from dataclasses import dataclass
from torch.utils.data import Dataset
import torch

@dataclass
class DemoDataset(Dataset):
    """Random dataset with 2-D inputs and 3-D targets."""

    size: int  # number of samples

    def __post_init__(self) -> None:
        self.x = torch.rand(self.size, 2)
        self.y = torch.rand(self.size, 3)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
