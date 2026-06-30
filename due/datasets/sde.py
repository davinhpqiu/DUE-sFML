import numpy as np
np.random.seed(0)
from scipy.io import loadmat

class sde_dataset():
    """
    A class representing an SDE dataset for stochastic flow map learning.

    Each trajectory in the .mat file has shape (N, d, T+1). All consecutive
    one-step transitions are extracted, with an optional memory window of m
    past states prepended to the input. Inputs and outputs are normalised to
    [-1, 1] using the same per-dimension scaling.

    With memory=0 (default), behaviour identical to original single-step-pair version:
        trainX: (J, d),          each row is x_n
        trainY: (J, d),          each row is x_{n+1}

    With memory=m > 0:
        trainX: (J, d*(m+1)),    each row is [x_{n-m}, ..., x_{n-1}, x_n] flattened
        trainY: (J, d),          each row is x_{n+1}

    Args:
        config (dict): dictionary containing configuration parameters.
            - problem_dim (int): The number of state variables d.
            - memory (int): Number of past states to include in the input (default 0).
            - dtype (str): The data type ('float32' or 'float64').

    Attributes:
        problem_dim (int): Number of state variables.
        memory_steps (int): Number of past states prepended to the input.
        dtype (str): Data type.

    Methods:
        load(file_path_train, file_path_test):
            Loads dataset from given file paths.

        normalize(data_X, data_Y):
            Normalises input and output data into range [-1, 1].
    """

    def __init__(self, config):
        self.problem_dim  = config["problem_dim"]
        self.memory_steps = config.get("memory", 0)
        self.dtype        = config["dtype"]

    def load(self, file_path_train, file_path_test):
        """
        Loads dataset from given file paths.

        Args:
            file_path_train (str): file path of training data.
            file_path_test (str or None): file path of test data.

        Returns:
            tuple:
                - trainX (ndarray): Normalised input data, shape (J, d*(memory+1)).
                - trainY (ndarray): Normalised output data, shape (J, d).
                - test_data (ndarray): Raw test trajectories, shape (N_test, d, T_test+1).
                - vmin (ndarray): Per-variable minimum values, shape (1, d).
                - vmax (ndarray): Per-variable maximum values, shape (1, d).
        """
        try:
            data = loadmat(file_path_train)
        except NotImplementedError:
            print("Your mat file is too large. Be patient.")
            import mat73
            data = mat73.loadmat(file_path_train)
        try:
            data = data["trajectories"]
        except:
            raise ValueError("Please name your dataset as trajectories.")

        N = data.shape[0] # number of trajectories
        T = data.shape[2] - 1 # number of time steps in each trajectory
        d = self.problem_dim
        m = self.memory_steps

        if data.shape[1] != d:
            raise ValueError(
                'Only support data arrays with size (N,d,T+1), '
                'N being number of trajectories, d being the number of state '
                'variables, T being the number of time steps.'
            )
        print("Dataset loaded, {} trajectories, {} variables, {} time instances".format(
            N, d, T + 1))

        # Extract one-step pairs with optional memory window
        # Valid starting positions: t = 0, 1, ..., T-m-1
        # Input window: data[:, :, t : t+m+1] → shape (N, d, m+1)
        # Output: data[:, :, t+m+1] → shape (N, d)
        # Total pairs: N * (T - m)
        n_pairs = T - m

        X_parts = []
        Y_parts = []
        for t in range(n_pairs):
            window = data[:, :, t : t + m + 1]          # (N, d, m+1)
            # Transpose to (N, m+1, d), flatten last two dims → (N, d*(m+1))
            X_parts.append(window.transpose(0, 2, 1).reshape(N, d * (m + 1)))
            Y_parts.append(data[:, :, t + m + 1])       # (N, d)

        target_X = np.concatenate(X_parts, axis=0)   # (N*n_pairs, d*(m+1))
        target_Y = np.concatenate(Y_parts, axis=0)   # (N*n_pairs, d)

        print("Dataset regrouped, {} pairs, {} variables".format(
            N * n_pairs, d))

        # Shuffle, normalise
        idx      = np.random.permutation(N * n_pairs)
        target_X = target_X[idx]
        target_Y = target_Y[idx]
        target_X, target_Y, vmin, vmax = self.normalize(target_X, target_Y)
        print("Training data is normalized")

        trainX = target_X
        trainY = target_Y
        print("Input shape {}.".format(trainX.shape),
              "Output shape {}.".format(trainY.shape))

        if file_path_test is None:
            return (trainX.astype(self.dtype), trainY.astype(self.dtype),
                    np.asarray(vmin).astype(self.dtype),
                    np.asarray(vmax).astype(self.dtype))
        else:
            try:
                test_raw = loadmat(file_path_test)
            except NotImplementedError:
                print("Your mat file is too large. Be patient.")
                import mat73
                test_raw = mat73.loadmat(file_path_test)
            test_data = test_raw["trajectories"]
            return (trainX.astype(self.dtype), trainY.astype(self.dtype),
                    test_data.astype(self.dtype),
                    np.asarray(vmin).astype(self.dtype),
                    np.asarray(vmax).astype(self.dtype))

    def normalize(self, data_X, data_Y):
        """
        Normalise input and output data into the range [-1, 1].

        All state values (across every memory step in X and in Y) share the
        same per-dimension scaling, so the same physical units map to the same
        normalised range everywhere.

        Args:
            data_X (ndarray): Input data, shape (J, d*(memory+1)).
            data_Y (ndarray): Output data, shape (J, d).

        Returns:
            tuple:
                - target_X (ndarray): Normalised input.
                - target_Y (ndarray): Normalised output.
                - vmin (ndarray): Per-variable minimum, shape (1, d).
                - vmax (ndarray): Per-variable maximum, shape (1, d).
        """
        d = self.problem_dim
        m = self.memory_steps

        # Reshape X to compute per-dimension stats across all time steps
        # (J, d*(m+1)) -> (J*(m+1), d)
        X_states = data_X.reshape(-1, d)

        # Joint min/max over all state values seen in both input and output
        vmax = np.maximum(np.max(X_states, axis=0, keepdims=True),
                          np.max(data_Y,   axis=0, keepdims=True))   # (1, d)
        vmin = np.minimum(np.min(X_states, axis=0, keepdims=True),
                          np.min(data_Y,   axis=0, keepdims=True))   # (1, d)

        # Broadcast vmin/vmax across all (m+1) memory steps for X
        vmax_X = np.tile(vmax, (1, m + 1)) # (1, d*(m+1))
        vmin_X = np.tile(vmin, (1, m + 1)) # (1, d*(m+1))

        target_X = 2 * (data_X - 0.5 * (vmax_X + vmin_X)) / (vmax_X - vmin_X)
        target_Y = 2 * (data_Y - 0.5 * (vmax   + vmin  )) / (vmax   - vmin  )
        target_X = np.clip(target_X, -1, 1)
        target_Y = np.clip(target_Y, -1, 1)

        return target_X, target_Y, vmin, vmax
