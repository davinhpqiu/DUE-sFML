import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import savetxt


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
        self.opt_G = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.opt_C = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=(beta1, beta2)
        )

        states = torch.cat([self.trainX.unsqueeze(-1), self.trainY], dim=-1)
        self.real_increments = states[..., 1:] - states[..., :-1]

        dataset = torch.utils.data.TensorDataset(self.trainX, self.real_increments)
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.bsize, shuffle=True, drop_last=True
        )

        # Columns:
        # critic_loss, generator_loss, wasserstein_gap, gp, score_real,
        # score_fake, real_increment_std, fake_increment_std,
        # increment_std_rel_error, checkpoint_selection_score
        self.hist = torch.full((self.nepochs, 10), float("nan"))
        self.critic_steps = 0

    def _advance_window(self, x_window, next_state):
        if x_window.shape[1] == self.output_dim:
            return next_state
        return torch.cat((x_window[..., self.output_dim:], next_state), dim=-1)

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
            z = torch.randn(
                x_window.size(0),
                self.latent_dim,
                device=self.device,
                dtype=x_window.dtype,
            )
            current_state = x_window[..., -self.output_dim:]
            det_next = self.det_net(x_window)
            stochastic_increment = self.generator(x_window, z)
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

    def checkpoint_score(self, epoch_values):
        real_std = epoch_values[6]
        fake_std = epoch_values[7]
        wgap = epoch_values[2]
        gp = epoch_values[3]
        std_rel_error = abs(fake_std - real_std) / (abs(real_std) + 1e-12)
        score = (
            self.selection_std_weight * std_rel_error
            + self.selection_wgap_weight * abs(wgap)
            + self.selection_gp_weight * gp
        )
        return std_rel_error, score

    def save_checkpoint(self, epoch, epoch_values):
        if self.checkpoint_interval <= 0:
            return

        epoch_num = epoch + 1
        std_rel_error, score = self.checkpoint_score(epoch_values)

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
            }
            critic_updates = 0
            generator_updates = 0

            for x0_batch, y_real_batch in self.train_loader:
                x0_batch = x0_batch.to(self.device)
                y_real_batch = y_real_batch.to(self.device)

                with torch.no_grad():
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

                if self.critic_steps % self.n_critic == 0:
                    y_fake_for_g = self.generate_increment_sequence(x0_batch)
                    score_fake_for_g = self.critic(x0_batch, y_fake_for_g).mean()
                    loss_G = -score_fake_for_g

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
            ]
            std_rel_error, selection_score = self.checkpoint_score(epoch_values)
            epoch_values.extend([std_rel_error, selection_score])
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
                    f"--- updates C/G: {critic_updates}/{generator_updates}"
                )
                start = end

            if self.checkpoint_interval > 0 and (ep + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(ep, epoch_values)

        torch.save(self.generator, self.save_path + "/generator_final")
        torch.save(self.critic, self.save_path + "/critic_final")
        if self.best_checkpoint_epoch is None:
            self.save_checkpoint(self.nepochs - 1, self.hist[-1].tolist())

    def save_hist(self, xlog=False, ylog=False):
        header = (
            "critic_loss generator_loss wasserstein_gap gradient_penalty "
            "score_real score_fake real_increment_std fake_increment_std "
            "increment_std_rel_error checkpoint_selection_score"
        )
        savetxt(self.save_path + "/training_history_gan.csv", self.hist.numpy(), header=header)

        hist = self.hist.numpy()
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, self.nepochs + 1), hist[:, 0], label="Critic loss")
        plt.plot(range(1, self.nepochs + 1), hist[:, 1], label="Generator loss")
        plt.plot(range(1, self.nepochs + 1), hist[:, 2], label="Wasserstein gap")
        plt.plot(range(1, self.nepochs + 1), hist[:, 3], label="Gradient penalty")
        plt.plot(range(1, self.nepochs + 1), hist[:, 9], label="Selection score")
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
        print("Sequence length: ", self.sequence_length)
        print("n_critic:        ", self.n_critic)
        print("GP lambda:       ", self.gp_lambda)
        print("Batches / epoch: ", len(self.train_loader))
        print("Checkpoint every:", self.checkpoint_interval, "epochs")
        print("The model is trained on", self.device)

    def set_seed(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
