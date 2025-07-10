import torch
from torch.utils.data import DataLoader
from cmix.data.demo_dataset import DemoDataset


def test_dataset_shapes():
    ds = DemoDataset(size=5)
    assert len(ds) == 5
    x, y = ds[0]
    assert x.shape == (2,)
    assert y.shape == (3,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert torch.all((0 <= x) & (x <= 1))
    assert torch.all((0 <= y) & (y <= 1))


def test_dataloader_batch():
    ds = DemoDataset(size=8)
    loader = DataLoader(ds, batch_size=4)
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (4, 2)
    assert y_batch.shape == (4, 3)
