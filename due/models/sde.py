import os
import torch
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
from time import time
from ..utils import *

class SDE:
    """
    Class representing the stochastic flow map learning (sFML) model.

    Implements Phase 2 of the sFML framework (Chen & Xiu, 2024):
    given a frozen deterministic sub-map D_theta,
    train a Generator G_phi and Critic C_psi via WGAN-GP so that
    G_phi(x_n, z) approximates draws from the residual distribution p(r | x_n), where r_n = x_{n+1} - D_theta(x_n).

    The full stochastic prediction at inference time is:
        x_{n+1} = D_theta(x_n) + G_phi(x_n, z),   z ~ N(0, I)

    Args:
        trainX (numpy array): Normalized input states x_n, shape (J, d).
        trainY (numpy array): Normalized output states x_{n+1}, shape (J, d).
        det_net: Trained and frozen deterministic sub-map D_theta (e.g., due.networks.fcn.resnet).
        generator: Generator network G_phi (due.networks.gan.Generator).
        critic: Critic network C_psi (due.networks.gan.Critic).
        config (dict): Configuration parameters for training.
            - device (str): 'cpu' or 'cuda'.
            - epochs (int): Number of training epochs.
            - batch_size (int): Batch size.
            - n_critic (int): Number of critic updates per generator update.
            - gp_lambda (float): Gradient penalty weight.
            - learning_rate (float): Learning rate for Adam optimizers.
            - adam_beta1 (float): Adam beta_1 parameter.
            - adam_beta2 (float): Adam beta_2 parameter.
            - latent_dim (int): Dimension of the latent noise vector z.
            - verbose (int): Print frequency (in epochs).
            - save_path (str): Directory to save the trained model and history.
            - seed (int): Random seed.

    Attributes:
        trainX (torch.Tensor): Normalized input states.
        trainY (torch.Tensor): Normalized output states.
        residuals (torch.Tensor): Precomputed residuals r_n = x_{n+1} - D_theta(x_n).
        det_net: Frozen deterministic sub-map.
        generator: Generator network.
        critic: Critic network.
        hist (torch.Tensor): Training history, shape (epochs, 2).
            Column 0: critic loss per epoch.
            Column 1: generator loss per epoch.

    Methods:
        train(): Runs the WGAN-GP training loop.
        gradient_penalty(x, r_real, r_fake): Computes the gradient penalty term.
        save_hist(xlog=False, ylog=False): Saves training history to CSV and PNG.
        summary(): Prints a summary of the model.
        set_seed(seed): Sets the random seed for reproducibility.
    """

    def __init__(self, trainX, trainY, det_net, generator, critic, config):
        super().__init__()

        # Seed
        self.set_seed(config["seed"])
        self.device = config["device"]

        # Keep data as CPU tensors; moved to device per-batch inside train()
        self.trainX = torch.from_numpy(trainX)
        self.trainY = torch.from_numpy(trainY)

        # Freeze D_theta
        self.det_net = det_net.to(self.device)
        for param in self.det_net.parameters():
            param.requires_grad = False
        self.det_net.eval()

        # Precompute residuals r_n = x_{n+1} - D_theta(x_n)
        print("Precomputing residuals r_n = x_{n+1} - D_theta(x_n) ...")
        with torch.no_grad():
            X_dev = self.trainX.to(self.device)
            Y_dev = self.trainY.to(self.device)
            self.residuals = (Y_dev - self.det_net(X_dev)).cpu()
        print("Residuals computed. Shape:", self.residuals.shape)

        # Move Generator and Critic to device; weights updated during training
        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)

        # hyperparameters from config
        self.nepochs = config["epochs"] # total training epochs
        self.bsize = config["batch_size"] # samples per batch
        self.n_critic = config["n_critic"] # critic steps per generator step (paper: 5)
        self.gp_lambda = config["gp_lambda"] # weight on gradient penalty term (paper: 10)
        self.latent_dim = config["latent_dim"] # dimension of z ~ N(0, I)
        self.verbose = config["verbose"] # print every this many epochs
        self.save_path = config["save_path"] # directory for saved models and plots

        # save directory
        try:
            os.mkdir(self.save_path)
        except:
            pass

        # Optimisers
        # Separate Adam instances for Generator and Critic

        lr = config["learning_rate"]
        beta1 = config["adam_beta1"]
        beta2 = config["adam_beta2"]
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_C = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(beta1, beta2))

        # Pre-allocate history: (epochs, 2) — col 0 = critic loss, col 1 = generator loss
        self.hist = torch.zeros(self.nepochs, 2)

        # DataLoader 
        # TensorDataset zips (x_n, r_n) matched pairs.
        # shuffle=True re-randomises order each epoch
        # drop_last=True discards final incomplete batch
        dataset = torch.utils.data.TensorDataset(self.trainX, self.residuals)
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.bsize, shuffle=True, drop_last=True
        )

    def gradient_penalty(self, x, r_real, r_fake):
        """
        Computes the WGAN-GP gradient penalty.

        Interpolates between real and fake residuals, evaluates the critic,
        penalises deviation of the gradient norm from 1.

        The GP enforces the 1-Lipschitz constraint on the critic,
        required for the Wasserstein distance estimate to be valid. 
        Without it, the critic can grow unboundedly and training diverges.

        Args:
            x (torch.Tensor): State batch, shape (batch, d).
            r_real (torch.Tensor): Real residuals from data, shape (batch, d).
            r_fake (torch.Tensor): Fake residuals from Generator, shape (batch, d).

        Returns:
            torch.Tensor: Scalar gradient penalty.
        """
        batch = r_real.size(0)

        # One interpolation weight per sample from U(0,1)
        # Shape (batch, 1) broadcasts against (batch, d) residuals

        alpha = torch.rand(batch, 1, device=self.device, dtype=r_real.dtype)

        # convex combination of real and fake residuals
        # .detach() removes r_fake from the generator's graph
        # .requires_grad_(True) attaches fresh tracker to differentiate the critic score wrt r_hat

        r_hat = (alpha * r_real + (1 - alpha) * r_fake.detach()).requires_grad_(True)

        score = self.critic(x, r_hat)

        # Compute d(score)/d(r_hat) for each sample in the batch.
        # grad_outputs=ones_like(score): score is (batch,1), sums before differentiating, one gradient vector per sample
        # create_graph=True: builds graph so loss_C.backward() differentiate GP term when updating critic weights
        # retain_graph=True: keeps intermediate graph for reuse
        # [0]: autograd.grad returns a tuple; unpack the single result
        grad = torch.autograd.grad(
            outputs=score,
            inputs=r_hat,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )[0]

        # GP = E[(||grad|| - 1)^2]: penalises any deviation of the gradient norm from 1.

        # For OU d=1, norm is absolute value of the scalar gradient.
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def train(self):
        """
        Runs the WGAN-GP training loop.

        For each epoch, iterates over batches. Each batch performs n_critic
        critic updates (with gradient penalty) followed by 1 generator update.

        Saves generator_final and critic_final after all epochs complete.
        Use these for evaluation.
        """
        self.summary()

        overal_start = time()
        start = overal_start

        for ep in range(self.nepochs):
            # Training mode: enables dropout/batchnorm if present 
            self.generator.train()
            self.critic.train()

            # Accumulators for per-epoch average losses
            epoch_critic_loss = 0.
            epoch_gen_loss    = 0.
            n_batches = 0

            for x_batch, r_batch in self.train_loader:
                # Move batch to device 
                x_batch = x_batch.to(self.device)
                r_batch = r_batch.to(self.device)


                # CRITIC UPDATE: n_critic times per batch (paper: 5)
                # Loss = E[C(fake)] - E[C(real)] + lambda * GP
                # Minimising loss pushes real scores up, fake scores down,

                for _ in range(self.n_critic):
                    # Fresh z each critic step — shape (batch, latent_dim)
                    z = torch.randn(x_batch.size(0), self.latent_dim,
                                    device=self.device, dtype=x_batch.dtype)

                    # .detach() cuts the generator graph
                    # gradient must not flow into generator during critic update step

                    r_fake = self.generator(x_batch, z).detach()

                    gp = self.gradient_penalty(x_batch, r_batch, r_fake)
                    score_real = self.critic(x_batch, r_batch).mean() 
                    score_fake = self.critic(x_batch, r_fake).mean()

                    # WGAN-GP critic loss
                    loss_C = score_fake - score_real + self.gp_lambda * gp

                    self.opt_C.zero_grad() # clear stale gradients from last step
                    loss_C.backward() # d(loss_C)/d(critic weights)
                    self.opt_C.step() # Adam update — critic weights only


                # GENERATOR UPDATE — once per batch
                # Loss = -E[C(fake)]  (maximise critic score on fakes)

                # Fresh z
                # generator(x, z) so generator weights get updated
                z = torch.randn(x_batch.size(0), self.latent_dim,
                                device=self.device, dtype=x_batch.dtype)
                r_fake  = self.generator(x_batch, z)
                loss_G  = -self.critic(x_batch, r_fake).mean()

                self.opt_G.zero_grad() # clear stale gradients
                loss_G.backward() # d(loss_G)/d(generator weights)
                self.opt_G.step() # Adam update — generator weights only

                # .item() converts scalar tensor to Python float, detaches from graph
                epoch_critic_loss += loss_C.item()
                epoch_gen_loss += loss_G.item()
                n_batches += 1

            # Average over all batches (for OU: 400000/256 = 1562 batches/epoch)
            epoch_critic_loss /= n_batches
            epoch_gen_loss /= n_batches

            # Record in history tensor (col 0 = critic, col 1 = generator)
            self.hist[ep, 0] = epoch_critic_loss
            self.hist[ep, 1] = epoch_gen_loss

            # Print progress; measure wall time per verbose interval
            if (ep + 1) % self.verbose == 0:
                end = time()
                print(f"Epoch {ep+1} --- Time: {end-start:.2f} seconds --- Critic loss: {epoch_critic_loss:.6f} --- Generator loss: {epoch_gen_loss:.6f}")
                start = end


        torch.save(self.generator, self.save_path + "/generator_final")
        torch.save(self.critic,    self.save_path + "/critic_final")

    def save_hist(self, xlog=False, ylog=False):
        """
        Saves the training history to a CSV file and a PNG plot.

        Args:
            xlog (bool): Use log scale on x-axis.
            ylog (bool): Use log scale on y-axis.
        """
        # Write raw loss values as plain-text CSV: two columns (critic, generator)
        savetxt(self.save_path + "/training_history_gan.csv", self.hist.numpy())

        plt.figure(figsize=(9, 9))
        # hist[:, 0] = critic losses per epoch, hist[:, 1] = generator losses per epoch
        plt.plot(range(1, self.nepochs + 1), self.hist[:, 0].numpy(), label="Critic loss")
        plt.plot(range(1, self.nepochs + 1), self.hist[:, 1].numpy(), label="Generator loss")
        plt.legend()
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.xlabel("Epoch")
        plt.savefig(self.save_path + "/training_history_gan.png")
        plt.close()   # release figure memory; omitting this leaks figures over long pipelines

    def summary(self):
        """Prints model configuration before training begins."""
        print("Generator trainable parameters:", self.generator.count_params())
        print("Critic    trainable parameters:", self.critic.count_params())
        print()
        print("Number of epochs:", self.nepochs)
        print("Batch size:      ", self.bsize)
        print("n_critic:        ", self.n_critic)
        print("GP lambda:       ", self.gp_lambda)
        print("The model is trained on", self.device)

    def set_seed(self, seed):
        """
        Sets all random seeds for full reproducibility.
        Covers Python hashing, PyTorch CPU/GPU RNGs, and cuDNN algorithm selection.
        """
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # multi-GPU
        torch.backends.cudnn.benchmark = False # don't auto-select fastest conv algorithm
        torch.backends.cudnn.deterministic = True # force deterministic cuDNN ops
