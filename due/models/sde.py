import os
import torch
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
from time import time
from ..utils import *

class SDE:
    """
    Sequence-level WGAN-GP training for sFML (Algorithm 4.1, Chen & Xiu 2024).

    Trains Generator G_phi and Critic C_psi on full L-step trajectory sequences,
    matching the paper exactly:

      - Per training step, the generator is rolled out L steps from x_0 to produce
        a fake increment sequence ŷ_{1:L} where ŷ_j = x̂_j - x̂_{j-1}.
      - The critic discriminates (x_0, ŷ_{1:L}) vs (x_0, y_{1:L}) — it sees the
        full temporal structure, not just single-step pairs.
      - Gradient penalty is computed in the joint (x_0, y_{1:L}) space.

    At inference time the stochastic map is unchanged:
        x_{n+1} = D_theta(x_n) + G_phi(x_n, z),   z ~ N(0, I)

    Args:
        train_seqs (numpy array): Normalised trajectory sequences, shape (N, d, L+1).
            Each sequence contains L+1 time points starting from x_0.
        det_net: Trained and frozen deterministic sub-map D_theta.
        generator: Generator network G_phi (due.networks.gan.Generator).
        critic: Critic network C_psi (due.networks.gan.Critic).
        config (dict): Training configuration.
            - device (str): 'cpu' or 'cuda'.
            - epochs (int): Number of training epochs.
            - batch_size (int): Trajectories per batch.
            - n_critic (int): Critic updates per generator update (paper: 5).
            - gp_lambda (float): Gradient penalty weight (paper: 10).
            - learning_rate (float): Adam learning rate (paper: 5e-5).
            - adam_beta1 (float): Adam beta_1 (paper: 0.5).
            - adam_beta2 (float): Adam beta_2 (paper: 0.999).
            - latent_dim (int): Dimension of z ~ N(0, I).
            - verbose (int): Print every this many epochs.
            - save_path (str): Directory for saved models and history.
            - seed (int): Random seed.

    Attributes:
        x0 (Tensor): Normalised initial states, shape (N, d).
        y_real (Tensor): Real increment sequences, shape (N, d, L).
                         y_real[:, :, j] = x_{j+1} - x_j from training data.
        L (int): Sequence length (number of increments).
        hist (Tensor): Training history (epochs, 2) — col 0 critic, col 1 generator.
    """

    def __init__(self, train_seqs, det_net, generator, critic, config):
        super().__init__()

        self.set_seed(config["seed"])
        self.device = config["device"]

        # Unpack trajectory dimensions
        N, d, Lp1 = train_seqs.shape
        self.L = Lp1 - 1   # number of increment steps
        self.d = d

        # Convert to tensors (kept on CPU; moved to device per-batch in train())
        seqs = torch.from_numpy(train_seqs)
        self.x0     = seqs[:, :, 0]                          # (N, d) initial states
        self.y_real = seqs[:, :, 1:] - seqs[:, :, :-1]      # (N, d, L) real increments

        # Freeze D_theta — its parameters are never updated during Phase 2
        self.det_net = det_net.to(self.device)
        for param in self.det_net.parameters():
            param.requires_grad = False
        self.det_net.eval()

        self.generator = generator.to(self.device)
        self.critic    = critic.to(self.device)

        # Hyperparameters
        self.nepochs    = config["epochs"]
        self.bsize      = config["batch_size"]
        self.n_critic   = config["n_critic"]
        self.gp_lambda  = config["gp_lambda"]
        self.latent_dim = config["latent_dim"]
        self.verbose    = config["verbose"]
        self.save_path  = config["save_path"]

        try:
            os.mkdir(self.save_path)
        except:
            pass

        lr    = config["learning_rate"]
        beta1 = config["adam_beta1"]
        beta2 = config["adam_beta2"]
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_C = torch.optim.Adam(self.critic.parameters(),    lr=lr, betas=(beta1, beta2))

        # Learning-rate schedule. The GAN paper (Chen & Xiu 2022) decays lr from
        # 5e-5 to 1e-5 with cosine annealing; sFML inherits this. Applied per epoch
        # to both optimizers. Default lr_min = lr/5 reproduces 5e-5 -> 1e-5.
        self.sched_type = config.get("scheduler", "cosine")
        lr_min = float(config.get("learning_rate_min", lr / 5.0))
        if self.sched_type == "cosine":
            self.sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_G, T_max=self.nepochs, eta_min=lr_min)
            self.sched_C = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_C, T_max=self.nepochs, eta_min=lr_min)
        else:
            self.sched_G = self.sched_C = None
        self.lr_min = lr_min

        # History: col 0 = critic loss, col 1 = generator loss
        self.hist = torch.zeros(self.nepochs, 2)

        # DataLoader over trajectories — each item is (x0, y_real) for one trajectory
        dataset = torch.utils.data.TensorDataset(self.x0, self.y_real)
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.bsize, shuffle=True, drop_last=True
        )

    def _rollout(self, x0_batch):
        """
        Roll out the stochastic map L steps from x0 (Algorithm 4.1, lines 5-8).

        At each step j:
            z_j      ~ N(0, I)
            x̂_{j+1}  = D_theta(x̂_j) + G_phi(x̂_j, z_j)
            ŷ_{j+1}  = x̂_{j+1} - x̂_j   (state increment)

        Called inside torch.no_grad() for critic updates (no generator gradients
        needed) and with the full computation graph for the generator update.

        Args:
            x0_batch (Tensor): (B, d) normalised initial states.

        Returns:
            Tensor: (B, d*L) flattened fake increment sequence ŷ_{1:L}.
        """
        x = x0_batch
        increments = []
        for _ in range(self.L):
            z      = torch.randn(x.size(0), self.latent_dim, device=self.device, dtype=x.dtype)
            r      = self.generator(x, z)
            x_next = self.det_net(x) + r
            increments.append(x_next - x)   # ŷ_{j+1} = x̂_{j+1} - x̂_j
            x = x_next
        y_fake = torch.stack(increments, dim=2)    # (B, d, L)
        return y_fake.reshape(y_fake.size(0), -1)  # (B, d*L)

    def gradient_penalty(self, x0, y_real, y_fake):
        """
        WGAN-GP gradient penalty in the joint (x_0, y_{1:L}) space
        (Algorithm 4.1, lines 10-11).

        Interpolates between real and fake increment sequences, then penalises
        any deviation of the critic's gradient norm from 1 over the full
        (d + d*L)-dimensional critic input.

        Args:
            x0 (Tensor): (B, d) initial states (real data — not interpolated).
            y_real (Tensor): (B, d*L) real increment sequences.
            y_fake (Tensor): (B, d*L) fake increment sequences (detached).

        Returns:
            Tensor: Scalar gradient penalty.
        """
        B     = y_real.size(0)
        alpha = torch.rand(B, 1, device=self.device, dtype=y_real.dtype)

        # Convex combination of real and fake increment sequences (line 10)
        y_hat = alpha * y_real + (1 - alpha) * y_fake

        # Attach gradient trackers to both parts of the critic input
        # so the penalty covers the full (x_0, ỹ_{1:L}) space (line 11)
        x0_r    = x0.detach().requires_grad_(True)     # (B, d)
        y_hat_r = y_hat.detach().requires_grad_(True)  # (B, d*L)

        seq_input = torch.cat([x0_r, y_hat_r], dim=1)  # (B, d*(1+L))
        score     = self.critic(seq_input)

        grads = torch.autograd.grad(
            outputs=score,
            inputs=[x0_r, y_hat_r],
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )
        # Concatenate gradients over the full input dimension (B, d*(1+L))
        grad = torch.cat(grads, dim=1)
        gp   = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def train(self):
        """
        WGAN-GP training loop (Algorithm 4.1).

        Per batch:
          1. n_critic critic updates — each scores full L-step sequences.
          2. 1 generator update — backprop through the full L-step rollout.
        """
        self.summary()

        overall_start = time()
        start         = overall_start

        for ep in range(self.nepochs):
            self.generator.train()
            self.critic.train()

            epoch_critic_loss = 0.
            epoch_gen_loss    = 0.
            n_batches         = 0

            for x0_batch, y_real_batch in self.train_loader:
                x0_batch     = x0_batch.to(self.device)      # (B, d)
                y_real_batch = y_real_batch.to(self.device)  # (B, d, L)

                B = x0_batch.size(0)

                # Flatten real increments: (B, d, L) -> (B, d*L)
                y_real_flat = y_real_batch.reshape(B, -1)

                # ---- CRITIC UPDATE (n_critic times per batch) ----
                # Fresh fake sequence per critic step; no generator gradients needed
                for _ in range(self.n_critic):
                    with torch.no_grad():
                        y_fake_flat = self._rollout(x0_batch)

                    real_input = torch.cat([x0_batch, y_real_flat], dim=1)
                    fake_input = torch.cat([x0_batch, y_fake_flat], dim=1)

                    score_real = self.critic(real_input).mean()
                    score_fake = self.critic(fake_input).mean()
                    gp         = self.gradient_penalty(x0_batch, y_real_flat, y_fake_flat)

                    # WGAN-GP critic loss (equation 4.19)
                    loss_C = score_fake - score_real + self.gp_lambda * gp

                    self.opt_C.zero_grad()
                    loss_C.backward()
                    self.opt_C.step()

                # ---- GENERATOR UPDATE (once per batch) ----
                # Full rollout WITH computation graph — gradients flow through all L steps
                y_fake_flat = self._rollout(x0_batch)
                fake_input  = torch.cat([x0_batch.detach(), y_fake_flat], dim=1)
                loss_G      = -self.critic(fake_input).mean()  # equation 4.20

                self.opt_G.zero_grad()
                loss_G.backward()
                self.opt_G.step()

                epoch_critic_loss += loss_C.item()
                epoch_gen_loss    += loss_G.item()
                n_batches         += 1

            self.hist[ep, 0] = epoch_critic_loss / n_batches
            self.hist[ep, 1] = epoch_gen_loss    / n_batches

            # Cosine LR decay (per epoch), applied to both critic and generator.
            if self.sched_G is not None:
                self.sched_G.step()
                self.sched_C.step()

            if (ep + 1) % self.verbose == 0:
                end = time()
                print(f"Epoch {ep+1} --- Time: {end-start:.2f}s --- "
                      f"Critic: {self.hist[ep,0]:.6f} --- Gen: {self.hist[ep,1]:.6f}")
                start = end

        torch.save(self.generator, self.save_path + "/generator_final")
        torch.save(self.critic,    self.save_path + "/critic_final")

    def save_hist(self, xlog=False, ylog=False):
        """Save training loss history to CSV and PNG."""
        savetxt(self.save_path + "/training_history_gan.csv", self.hist.numpy())
        plt.figure(figsize=(9, 9))
        plt.plot(range(1, self.nepochs + 1), self.hist[:, 0].numpy(), label="Critic loss")
        plt.plot(range(1, self.nepochs + 1), self.hist[:, 1].numpy(), label="Generator loss")
        plt.legend()
        if xlog: plt.xscale("log")
        if ylog: plt.yscale("log")
        plt.xlabel("Epoch")
        plt.savefig(self.save_path + "/training_history_gan.png")
        plt.close()

    def summary(self):
        """Print model configuration before training begins."""
        print("Generator trainable parameters:", self.generator.count_params())
        print("Critic    trainable parameters:", self.critic.count_params())
        print()
        print("Number of epochs:", self.nepochs)
        print("Batch size:      ", self.bsize)
        print("n_critic:        ", self.n_critic)
        print("GP lambda:       ", self.gp_lambda)
        print("Sequence length: ", self.L)
        if self.sched_G is not None:
            print(f"LR schedule:       cosine (base->{self.lr_min:.0e})")
        else:
            print("LR schedule:       constant")
        print("The model is trained on", self.device)

    def set_seed(self, seed):
        """Set all random seeds for reproducibility."""
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True
