import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import savetxt

from ..utils import MuonAdamW


class SDE:
    """
    Sequence-level stochastic flow-map learner following Algorithm 4.1.

    The deterministic map D_delta is frozen. The stochastic sub-map S_delta is
    trained with WGAN-GP by recurrently generating a full increment sequence
    y_{1:L}. The critic scores the pair (x_0, y_{1:L}), matching the paper's
    discriminator input rather than independent one-step residuals.
    """

    def __init__(self, trainX, trainY, det_net, generator, critic, config):
        super().__init__()

        self.set_seed(config["seed"])
        self.device = config["device"]

        self.trainX = torch.from_numpy(trainX)
        self.trainY = torch.from_numpy(trainY)
        if self.trainY.ndim != 3:
            raise ValueError(
                "Sequence-level SDE training expects trainY with shape (N, d, L)."
            )

        self.output_dim = self.trainY.shape[1]
        self.sequence_length = self.trainY.shape[2]

        self.det_net = det_net.to(self.device)
        for param in self.det_net.parameters():
            param.requires_grad = False
        self.det_net.eval()

        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)

        self.nepochs = config["epochs"]
        self.bsize = config["batch_size"]
        self.n_critic = config["n_critic"]
        self.gp_lambda = config["gp_lambda"]
        self.latent_dim = config["latent_dim"]
        self.verbose = config["verbose"]
        self.save_path = config["save_path"]
        self.checkpoint_interval = config.get("checkpoint_interval", 100)
        self.selection_std_weight = config.get("selection_std_weight", 1.0)
        self.selection_wgap_weight = config.get("selection_wgap_weight", 1.0)
        self.selection_gp_weight = config.get("selection_gp_weight", 1.0)
        # --- additions on top of the sequence branch (default 0 => his behaviour) ---
        # selection_mean_weight: weight of the increment-mean (drift) error in the
        #   checkpoint selection score. His score only used std/wgap/gp, so it could
        #   pick a good-variance/bad-mean checkpoint. Setting this > 0 makes the
        #   selection consider the drift too.
        self.selection_mean_weight = config.get("selection_mean_weight", 0.0)
        # mmd_lambda: weight of a Gaussian-kernel MMD term added to the generator
        #   loss. MMD matches the whole increment distribution (mean+variance+shape),
        #   directly pinning the mean bias and over-dispersion the weak critic misses.
        self.mmd_lambda = config.get("mmd_lambda", 0.0)
        # center_generator: hard-center the stochastic sub-map S_delta so that its
        #   conditional mean is zero BY CONSTRUCTION (enforces Remark 4.1 instead of
        #   leaving it emergent). For each state we draw center_K noise samples,
        #   average the generator outputs, and subtract that per-state mean:
        #       S_c(x, z0) = S(x, z0) - (1/K) sum_k S(x, z_k).
        #   Applied in BOTH training and evaluation (evaluation mirror lives in the
        #   example's rollout). The mean is then owned solely by the deterministic
        #   map D_delta (MSE-trained), off the noise-floor-limited adversarial path.
        #   center_K is a Monte-Carlo accuracy setting, not a tuned knob.
        self.center_generator = config.get("center_generator", False)
        self.center_K = int(config.get("center_K", 16))
        # single_step_critic: when True the critic sees individual (x_t, Δx_t) pairs
        # instead of full L-step sequences. This gives the critic direct signal on
        # per-step conditional variance with no gradient attenuation through L steps.
        self.single_step_critic = config.get("single_step_critic", False)

        self.best_checkpoint_score = float("inf")
        self.best_checkpoint_epoch = None

        try:
            os.mkdir(self.save_path)
        except:
            pass
        self.checkpoint_dir = self.save_path + "/checkpoints"
        if self.checkpoint_interval > 0:
            try:
                os.mkdir(self.checkpoint_dir)
            except:
                pass

        lr = config["learning_rate"]
        beta1 = config["adam_beta1"]
        beta2 = config["adam_beta2"]
        # Phase-2 optimizer toggle: "adam" (default, WGAN-GP betas) or "muon"
        # (Muon on 2D weight matrices + AdamW on biases; needs torch >= 2.13).
        self.optimizer_name = config.get("optimizer", "adam")

        def _make_opt(module):
            if self.optimizer_name in ("muon", "Muon", "MUON"):
                if not hasattr(torch.optim, "Muon"):
                    raise ValueError("torch.optim.Muon requires PyTorch >= 2.13 "
                                     "(upgrade torch, or install a Muon backport).")
                return MuonAdamW(module, lr)
            return torch.optim.Adam(module.parameters(), lr=lr, betas=(beta1, beta2))

        self.opt_G = _make_opt(self.generator)
        self.opt_C = _make_opt(self.critic)
        print(f"Phase-2 optimizer: {self.optimizer_name}")
        # Phase-2 cosine LR decay (from the 2022 GAN paper): shrinking the step size
        # over training damps the minimax oscillation (Robbins-Monro), which the
        # un-centered generator's weakly-damped mean direction needs to converge.
        self.lr_decay = config.get("lr_decay", False)
        self.lr_min = config.get("lr_min", 1e-5)
        if self.lr_decay:
            self.sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_G, T_max=self.nepochs, eta_min=self.lr_min)
            self.sched_C = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_C, T_max=self.nepochs, eta_min=self.lr_min)
        else:
            self.sched_G = self.sched_C = None

        states = torch.cat([self.trainX.unsqueeze(-1), self.trainY], dim=-1)
        self.real_increments = states[..., 1:] - states[..., :-1]

        if self.single_step_critic:
            # Flatten all N*L consecutive (x_t, Δx_t) pairs from training sequences.
            # states: (N, d, L+1) → x_flat: (N*L, d)
            # real_increments: (N, d, L) → dy_flat: (N*L, d)
            x_flat  = states[..., :-1].permute(0, 2, 1).reshape(-1, self.output_dim)
            dy_flat = self.real_increments.permute(0, 2, 1).reshape(-1, self.output_dim)
            dataset = torch.utils.data.TensorDataset(x_flat, dy_flat)
        else:
            dataset = torch.utils.data.TensorDataset(self.trainX, self.real_increments)

        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.bsize, shuffle=True, drop_last=True
        )

        # Columns (13):
        # 0 critic_loss, 1 generator_loss, 2 wasserstein_gap, 3 gp, 4 score_real,
        # 5 score_fake, 6 real_increment_std, 7 fake_increment_std,
        # 8 real_increment_mean, 9 fake_increment_mean,
        # 10 increment_std_rel_error, 11 increment_mean_abs_error,
        # 12 checkpoint_selection_score
        self.hist = torch.full((self.nepochs, 13), float("nan"))
        self.critic_steps = 0

    def _advance_window(self, x_window, next_state):
        if x_window.shape[1] == self.output_dim:
            return next_state
        return torch.cat((x_window[..., self.output_dim:], next_state), dim=-1)

    def _stochastic_increment(self, x_window):
        """
        Draw the stochastic sub-map increment S_delta(x, z) for each state in the
        batch. When center_generator is on, hard-center it per state over center_K
        noise draws so the conditional mean is zero by construction.
        """
        B = x_window.size(0)
        if not self.center_generator:
            z = torch.randn(
                B, self.latent_dim, device=self.device, dtype=x_window.dtype
            )
            return self.generator(x_window, z)

        K = self.center_K
        # center_K noise samples per state -> (B*K, latent_dim); one shared x per K.
        z = torch.randn(B, K, self.latent_dim, device=self.device, dtype=x_window.dtype)
        x_rep = x_window.unsqueeze(1).expand(-1, K, -1).reshape(B * K, x_window.size(-1))
        out = self.generator(x_rep, z.reshape(B * K, self.latent_dim))
        out = out.reshape(B, K, self.output_dim)
        # Use the 0th draw as the sample; subtract the per-state MC mean over all K.
        return out[:, 0, :] - out.mean(dim=1)

    def generate_increment_sequence(self, x0):
        """
        Recurrently generate y_{1:L} from x_0.

        Algorithm 4.1 uses
            y_hat_{j+1} = D(x_hat_j) - x_hat_j + S(x_hat_j, z_j)
            x_hat_{j+1} = x_hat_j + y_hat_{j+1}.
        """
        x_window = x0
        increments = []
        for _ in range(self.sequence_length):
            current_state = x_window[..., -self.output_dim:]
            det_next = self.det_net(x_window)
            stochastic_increment = self._stochastic_increment(x_window)
            increment = det_next - current_state + stochastic_increment
            next_state = current_state + increment
            increments.append(increment)
            x_window = self._advance_window(x_window, next_state)

        return torch.stack(increments, dim=-1)

    def gradient_penalty(self, x0, y_real, y_fake):
        batch = y_real.size(0)
        alpha_shape = [batch] + [1] * (y_real.ndim - 1)
        alpha = torch.rand(alpha_shape, device=self.device, dtype=y_real.dtype)

        x_hat = x0.detach().requires_grad_(True)
        y_hat = (alpha * y_real + (1 - alpha) * y_fake.detach()).requires_grad_(True)
        score = self.critic(x_hat, y_hat)

        grad_x, grad_y = torch.autograd.grad(
            outputs=score,
            inputs=(x_hat, y_hat),
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )
        grad = torch.cat(
            [grad_x.reshape(batch, -1), grad_y.reshape(batch, -1)], dim=1
        )
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def _mmd2(self, y_fake, y_real):
        """
        Biased MMD^2 with a Gaussian kernel between the fake and real increment
        sequences (each sample is the flattened y_{1:L}). Bandwidth via the median
        heuristic (detached). Matches the whole distribution -> pins mean+variance.
        """
        x = y_fake.reshape(y_fake.size(0), -1)
        y = y_real.reshape(y_real.size(0), -1)
        dxx = torch.cdist(x, x).pow(2)
        dyy = torch.cdist(y, y).pow(2)
        dxy = torch.cdist(x, y).pow(2)
        with torch.no_grad():
            med = torch.median(torch.cat([dxx.reshape(-1), dyy.reshape(-1), dxy.reshape(-1)]))
            h2 = med.clamp_min(1e-12)
        kxx = torch.exp(-dxx / (2.0 * h2))
        kyy = torch.exp(-dyy / (2.0 * h2))
        kxy = torch.exp(-dxy / (2.0 * h2))
        return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()

    def checkpoint_score(self, epoch_values):
        real_std = epoch_values[6]
        fake_std = epoch_values[7]
        real_mean = epoch_values[8]
        fake_mean = epoch_values[9]
        wgap = epoch_values[2]
        gp = epoch_values[3]
        std_rel_error = abs(fake_std - real_std) / (abs(real_std) + 1e-12)
        # Increment-mean (drift) error, normalised by the increment std so it is on
        # a comparable scale to std_rel_error (the mean bias is tiny in raw units).
        mean_abs_error = abs(fake_mean - real_mean) / (abs(real_std) + 1e-12)
        score = (
            self.selection_std_weight * std_rel_error
            + self.selection_mean_weight * mean_abs_error
            + self.selection_wgap_weight * abs(wgap)
            + self.selection_gp_weight * gp
        )
        return std_rel_error, mean_abs_error, score

    def save_checkpoint(self, epoch, epoch_values):
        if self.checkpoint_interval <= 0:
            return

        epoch_num = epoch + 1
        std_rel_error, mean_abs_error, score = self.checkpoint_score(epoch_values)

        generator_path = self.checkpoint_dir + f"/generator_epoch_{epoch_num:05d}"
        critic_path = self.checkpoint_dir + f"/critic_epoch_{epoch_num:05d}"
        torch.save(self.generator, generator_path)
        torch.save(self.critic, critic_path)

        is_best = score < self.best_checkpoint_score
        if is_best:
            self.best_checkpoint_score = score
            self.best_checkpoint_epoch = epoch_num
            torch.save(self.generator, self.save_path + "/generator_best")
            torch.save(self.critic, self.save_path + "/critic_best")
            with open(self.save_path + "/best_checkpoint.txt", "w") as f:
                f.write(f"epoch: {epoch_num}\n")
                f.write(f"selection_score: {score:.12g}\n")
                f.write(f"increment_std_rel_error: {std_rel_error:.12g}\n")
                f.write(f"increment_mean_abs_error: {mean_abs_error:.12g}\n")
                f.write(f"wasserstein_gap: {epoch_values[2]:.12g}\n")
                f.write(f"gradient_penalty: {epoch_values[3]:.12g}\n")
                f.write(f"real_increment_std: {epoch_values[6]:.12g}\n")
                f.write(f"fake_increment_std: {epoch_values[7]:.12g}\n")
                f.write(f"generator_path: {generator_path}\n")
                f.write(f"critic_path: {critic_path}\n")

        print(
            f"Checkpoint epoch {epoch_num} saved"
            f" --- selection score: {score:.6f}"
            f" --- std rel err: {std_rel_error:.6f}"
            f" --- mean abs err: {mean_abs_error:.6f}"
            f" --- best epoch: {self.best_checkpoint_epoch}"
        )

    def train(self):
        self.summary()

        start = time()
        for ep in range(self.nepochs):
            self.generator.train()
            self.critic.train()

            totals = {
                "critic_loss": 0.0,
                "generator_loss": 0.0,
                "wasserstein_gap": 0.0,
                "gp": 0.0,
                "score_real": 0.0,
                "score_fake": 0.0,
                "real_increment_std": 0.0,
                "fake_increment_std": 0.0,
                "real_increment_mean": 0.0,
                "fake_increment_mean": 0.0,
            }
            critic_updates = 0
            generator_updates = 0

            for x0_batch, y_real_batch in self.train_loader:
                x0_batch = x0_batch.to(self.device)
                y_real_batch = y_real_batch.to(self.device)

                with torch.no_grad():
                    if self.single_step_critic:
                        # One-step fake: D(x_t) - x_t + S(x_t, z)
                        det_inc = self.det_net(x0_batch) - x0_batch
                        y_fake_batch = det_inc + self._stochastic_increment(x0_batch)
                    else:
                        y_fake_batch = self.generate_increment_sequence(x0_batch)

                gp = self.gradient_penalty(x0_batch, y_real_batch, y_fake_batch)
                score_real = self.critic(x0_batch, y_real_batch).mean()
                score_fake = self.critic(x0_batch, y_fake_batch).mean()
                loss_C = score_fake - score_real + self.gp_lambda * gp

                self.opt_C.zero_grad()
                loss_C.backward()
                self.opt_C.step()

                self.critic_steps += 1
                critic_updates += 1

                totals["critic_loss"] += loss_C.item()
                totals["wasserstein_gap"] += (score_real - score_fake).item()
                totals["gp"] += gp.item()
                totals["score_real"] += score_real.item()
                totals["score_fake"] += score_fake.item()
                totals["real_increment_std"] += y_real_batch.std().item()
                totals["fake_increment_std"] += y_fake_batch.std().item()
                totals["real_increment_mean"] += y_real_batch.mean().item()
                totals["fake_increment_mean"] += y_fake_batch.mean().item()

                if self.critic_steps % self.n_critic == 0:
                    if self.single_step_critic:
                        with torch.no_grad():
                            det_inc = self.det_net(x0_batch) - x0_batch
                        y_fake_for_g = det_inc + self._stochastic_increment(x0_batch)
                    else:
                        y_fake_for_g = self.generate_increment_sequence(x0_batch)
                    score_fake_for_g = self.critic(x0_batch, y_fake_for_g).mean()
                    loss_G = -score_fake_for_g
                    # Optional MMD regulariser: match the whole increment distribution
                    # (mean + variance + shape) to the real batch. Off when lambda=0.
                    if self.mmd_lambda > 0.0:
                        loss_G = loss_G + self.mmd_lambda * self._mmd2(y_fake_for_g, y_real_batch)

                    self.opt_G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()

                    totals["generator_loss"] += loss_G.item()
                    generator_updates += 1

            c_div = max(critic_updates, 1)
            g_div = max(generator_updates, 1)
            epoch_values = [
                totals["critic_loss"] / c_div,
                totals["generator_loss"] / g_div if generator_updates > 0 else float("nan"),
                totals["wasserstein_gap"] / c_div,
                totals["gp"] / c_div,
                totals["score_real"] / c_div,
                totals["score_fake"] / c_div,
                totals["real_increment_std"] / c_div,
                totals["fake_increment_std"] / c_div,
                totals["real_increment_mean"] / c_div,
                totals["fake_increment_mean"] / c_div,
            ]
            std_rel_error, mean_abs_error, selection_score = self.checkpoint_score(epoch_values)
            epoch_values.extend([std_rel_error, mean_abs_error, selection_score])
            self.hist[ep] = torch.tensor(epoch_values)

            if (ep + 1) % self.verbose == 0:
                end = time()
                gen_msg = (
                    f"{epoch_values[1]:.6f}"
                    if generator_updates > 0
                    else "n/a"
                )
                print(
                    f"Epoch {ep+1} --- Time: {end-start:.2f}s "
                    f"--- C loss: {epoch_values[0]:.6f} "
                    f"--- G loss: {gen_msg} "
                    f"--- W gap: {epoch_values[2]:.6f} "
                    f"--- GP: {epoch_values[3]:.6f} "
                    f"--- C(real): {epoch_values[4]:.6f} "
                    f"--- C(fake): {epoch_values[5]:.6f} "
                    f"--- dy std real/fake: {epoch_values[6]:.6f}/{epoch_values[7]:.6f} "
                    f"--- dy mean real/fake: {epoch_values[8]:.6f}/{epoch_values[9]:.6f} "
                    f"--- updates C/G: {critic_updates}/{generator_updates}"
                )
                start = end

            if self.checkpoint_interval > 0 and (ep + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(ep, epoch_values)

            if self.sched_G is not None:
                self.sched_G.step()
                self.sched_C.step()

        torch.save(self.generator, self.save_path + "/generator_final")
        torch.save(self.critic, self.save_path + "/critic_final")
        if self.best_checkpoint_epoch is None:
            self.save_checkpoint(self.nepochs - 1, self.hist[-1].tolist())

    def save_hist(self, xlog=False, ylog=False):
        header = (
            "critic_loss generator_loss wasserstein_gap gradient_penalty "
            "score_real score_fake real_increment_std fake_increment_std "
            "real_increment_mean fake_increment_mean "
            "increment_std_rel_error increment_mean_abs_error checkpoint_selection_score"
        )
        savetxt(self.save_path + "/training_history_gan.csv", self.hist.numpy(), header=header)

        hist = self.hist.numpy()
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, self.nepochs + 1), hist[:, 0], label="Critic loss")
        plt.plot(range(1, self.nepochs + 1), hist[:, 1], label="Generator loss")
        plt.plot(range(1, self.nepochs + 1), hist[:, 2], label="Wasserstein gap")
        plt.plot(range(1, self.nepochs + 1), hist[:, 3], label="Gradient penalty")
        plt.plot(range(1, self.nepochs + 1), hist[:, 12], label="Selection score")
        plt.legend()
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(self.save_path + "/training_history_gan.png")
        plt.close()

    def summary(self):
        print("Generator trainable parameters:", self.generator.count_params())
        print("Critic    trainable parameters:", self.critic.count_params())
        print()
        print("Number of epochs:", self.nepochs)
        print("Batch size:      ", self.bsize)
        if self.single_step_critic:
            print("Critic mode:      single-step (x_t, Δx_t) pairs — no rollout")
        else:
            print("Critic mode:      sequence (L=%d steps)" % self.sequence_length)
        print("n_critic:        ", self.n_critic)
        print("GP lambda:       ", self.gp_lambda)
        print("Batches / epoch: ", len(self.train_loader))
        print("Checkpoint every:", self.checkpoint_interval, "epochs")
        if self.center_generator:
            print(f"Hard-centering:   ON (zero-mean S_delta, K={self.center_K})")
        else:
            print("Hard-centering:   OFF")
        if self.lr_decay:
            print(f"LR decay:         ON (cosine {self.opt_G.param_groups[0]['lr']:.1e} -> {self.lr_min:.1e})")
        else:
            print("LR decay:         OFF (constant lr)")
        if self.mmd_lambda > 0.0:
            print(f"MMD term:         ON (lambda={self.mmd_lambda})")
        print("The model is trained on", self.device)

    def set_seed(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
