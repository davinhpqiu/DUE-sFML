import torch
# DUE library convention set float64 as default dtype globally.
# Hence torch.nn.Linear() without explicit dtype produces float64 weights.
# In 'single' branch below call .float() on every layer to override.
torch.set_default_dtype(torch.float64)
from .nn import nn
from ..utils import get_activation

class Generator(nn):
    """
    Generator network G_phi for stochastic flow map learning.

    Maps a state x_n and a latent noise vector z to a residual sample r_tilde,
    approximating draws from the conditional noise distribution p(r | x_n):

        r_tilde = G_phi(x_n, z),   z ~ N(0, I_{latent_dim})

    During stochastic prediction, the full one-step map is:

        x_{n+1} = D_theta(x_n) + G_phi(x_n, z)

    Args:
        config (dict): A dictionary containing configuration parameters.
            - problem_dim (int): The number of state variables d.
            - latent_dim (int): The dimension of the latent noise vector z.
            - memory (int): Number of past states prepended to the state input (default 0).
            - condition_on_state (bool): If True, Generator receives (x, z); if False, only z.
            - depth (int): The number of hidden layers.
            - width (int): The number of neurons in each hidden layer.
            - activation (str): The activation function name.
            - dtype (str): The data type ('single' or 'double').
            - seed (int): The random seed.

    Attributes:
        output_dim (int): Dimension of the output (equals problem_dim).
        input_dim (int): Dimension of the network input.
        latent_dim (int): Dimension of the latent noise vector.
        memory_steps (int): Number of past states in the input window.
        condition_on_state (bool): Whether x is included in the input.
        depth (int): Number of hidden layers.
        width (int): Number of neurons per hidden layer.
        activation (function): Activation function.
        layers (torch.nn.ModuleList): List of linear layers.

    Methods:
        forward(x, z): Computes r_tilde = G_phi(concat(x, z)) or G_phi(z).
        sample(x, n_samples, device): Draws n_samples residuals from the same x.
    """

    def __init__(self, config):
        super().__init__()

        # Output d-dimensional residual, same shape as state
        self.output_dim = config["problem_dim"]

        # z drawn from N(0, I_latent_dim)
        self.latent_dim = config["latent_dim"]

        # memory_steps = m means the input includes [x_{n-m}, ..., x_{n-1}, x_n],
        # flattened to a vector of length d*(m+1). With m=0 (default), input is just x_n.
        self.memory_steps = config.get("memory", 0)

        # condition_on_state = True  (default): input is concat(x_window, z)
        # condition_on_state = False (unconditional):  input is z only — Generator
        self.condition_on_state = config.get("condition_on_state", True)

        # Compute network input dimension
        # state_input_dim = d*(m+1): full state window flattened.
        state_input_dim = self.output_dim * (self.memory_steps + 1)
        self.input_dim  = (state_input_dim + self.latent_dim
                           if self.condition_on_state else self.latent_dim)

        self.depth = config["depth"] # number of hidden layers
        self.width = config["width"] # neurons per hidden layer
        self.activation = get_activation(config["activation"]) # activation function
        self.dtype = config["dtype"] # single or double precision

        self.set_seed(config["seed"])

        self.layers = torch.nn.ModuleList()

        if self.dtype == "double":
            # First hidden layer: projects from input_dim up to width
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(torch.nn.Linear(self.input_dim, self.width).double())
                else:
                    # Remaining hidden layers: width -> width (same shape)
                    self.layers.append(torch.nn.Linear(self.width, self.width).double())
            # Output layer: projects from width down to output_dim (residual dimension).
            self.layers.append(torch.nn.Linear(self.width, self.output_dim).double())

        elif self.dtype == "single":
            # .float() overrides the global float64 default, identical structure
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(torch.nn.Linear(self.input_dim, self.width).float())
                else:
                    self.layers.append(torch.nn.Linear(self.width, self.width).float())
            self.layers.append(torch.nn.Linear(self.width, self.output_dim).float())
        else:
            print("self.dtype error. The self.dtype must be either single or double.")
            exit()

    def forward(self, x, z):
        """
        Forward pass through the Generator.

        Args:
            x (torch.Tensor): Normalised state window, shape (batch, d*(memory+1)).
                              Ignored when condition_on_state is False.
            z (torch.Tensor): Latent noise sample, shape (batch, latent_dim).
                              Drawn fresh from N(0,I) at every call during prediction.

        Returns:
            torch.Tensor: Fake residual r_tilde, shape (batch, d).
        """
        # network input
        inp = torch.cat([x, z], dim=-1) if self.condition_on_state else z

        # Pass through all hidden layers with activation
        for l in self.layers[:-1]:
            inp = self.activation(l(inp))

        # Final linear layer with no activation — output raw residual r_tilde.
        return self.layers[-1](inp)

    def sample(self, x, n_samples, device):
        """
        Draw n_samples residual samples from the Generator for a single state x.

        NOT used during training. Used at evaluation time to build an ensemble
        of residuals from a fixed state, approximating p(r | x_n).

        Args:
            x (torch.Tensor): A single normalised state, shape (d,) or (1, d).
            n_samples (int): Number of independent residual samples to draw.
            device (str): Device string ('cpu', 'cuda', etc.).

        Returns:
            torch.Tensor: Residual samples, shape (n_samples, d).
        """
        self.to(device)
        self.eval()

        # Broadcast the single state x to a batch of n_samples identical rows.
        x = x.reshape(1, -1).expand(n_samples, -1).to(device)

        # Draw n_samples independent noise vectors — each gives a different residual.
        z = torch.randn(n_samples, self.latent_dim, device=device, dtype=x.dtype)

        with torch.no_grad():
            return self.forward(x, z)


class Critic(nn):
    """
    Critic network C_psi for Wasserstein GAN training.

    Maps a state and either a one-step residual or a residual sequence to a raw
    scalar score:

        score = C_psi(x_0, y_{1:L})

    The critic is trained to maximise the gap between scores for real and fake residuals. 
    The generator is trained to make fake residuals indistinguishable from real ones.

    Args:
        config (dict): A dictionary containing configuration parameters.
            - problem_dim (int): The number of state variables d.
            - memory (int): Number of past states in the state input (default 0).
            - sequence_length (int): Number of increments scored by the critic.
            - condition_on_state (bool): If True, Critic receives (x, y); if False, only y.
            - depth (int): The number of hidden layers.
            - width (int): The number of neurons in each hidden layer.
            - activation (str): The activation function name.
            - dtype (str): The data type ('single' or 'double').
            - seed (int): The random seed.

    Attributes:
        output_dim (int): Always 1 — a scalar Wasserstein score.
        input_dim (int): Dimension of the network input.
        memory_steps (int): Number of past states in the input window.
        condition_on_state (bool): Whether x is included in the input.
        depth (int): Number of hidden layers.
        width (int): Number of neurons per hidden layer.
        activation (function): Activation function.
        layers (torch.nn.ModuleList): List of linear layers.

    Methods:
        forward(x, r): Computes the scalar score C_psi(concat(x, r)) or C_psi(r).
    """

    def __init__(self, config):
        super().__init__()

        self.output_dim = 1

        # Same memory convention as the Generator
        self.memory_steps = config.get("memory", 0)

        # condition_on_state = True (paper): input is concat(x_window, r)
        # condition_on_state = False (unconditional):  input is r only
        self.condition_on_state = config.get("condition_on_state", True)

        self.problem_dim = config["problem_dim"]
        self.sequence_length = config.get("sequence_length", 1)

        # State input d*(memory+1). Increment sequence y_{1:L} has d*L entries.
        state_input_dim = config["problem_dim"] * (self.memory_steps + 1)
        sequence_input_dim = config["problem_dim"] * self.sequence_length
        self.input_dim  = (state_input_dim + sequence_input_dim
                           if self.condition_on_state else sequence_input_dim)

        self.depth = config["depth"]
        self.width = config["width"]
        self.activation = get_activation(config["activation"])
        self.dtype = config["dtype"]

        self.set_seed(config["seed"])
        self.layers = torch.nn.ModuleList()

        if self.dtype == "double":
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(torch.nn.Linear(self.input_dim, self.width).double())
                else:
                    self.layers.append(torch.nn.Linear(self.width, self.width).double())
            # Output layer maps to a single scalar — the Wasserstein score
            self.layers.append(torch.nn.Linear(self.width, self.output_dim).double())

        elif self.dtype == "single":
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(torch.nn.Linear(self.input_dim, self.width).float())
                else:
                    self.layers.append(torch.nn.Linear(self.width, self.width).float())
            self.layers.append(torch.nn.Linear(self.width, self.output_dim).float())
        else:
            print("self.dtype error. The self.dtype must be either single or double.")
            exit()

    def forward(self, x, y):
        """
        Forward pass through the Critic.

        Called during the critic update with real and generated increment
        sequences. Called during the generator update with generated increments
        only, where the generator tries to increase the critic score.

        Args:
            x (torch.Tensor): Normalised initial state/window, shape (batch, d*(memory+1)).
                              Ignored when condition_on_state is False.
            y (torch.Tensor): Increment sequence to score. Shape can be
                              (batch, d) for one-step training or (batch, d, L)
                              for Algorithm 4.1 sequence training.

        Returns:
            torch.Tensor: Raw Wasserstein score, shape (batch, 1).
                          Higher = critic judges y as more likely real.
                          No sigmoid — this is an unbounded real number.
        """
        y_flat = y.reshape(y.shape[0], -1)
        inp = torch.cat([x, y_flat], dim=-1) if self.condition_on_state else y_flat

        # Hidden layers with activation
        for l in self.layers[:-1]:
            inp = self.activation(l(inp))

        return self.layers[-1](inp)
