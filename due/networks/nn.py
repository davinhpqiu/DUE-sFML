import os

import torch


class nn(torch.nn.Module):
    """Base class for DUE neural network modules."""

    def __init__(self):
        super().__init__()

    def count_params(self):
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

    def load_params(self, save_path):
        return torch.load(save_path)

    def set_seed(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
