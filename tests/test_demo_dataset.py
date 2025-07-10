import jax.numpy as jnp

from cmix.data.demo_dataset import DataLoader, DemoDataset


def test_dataset_shapes():
    ds = DemoDataset(size=5, seed=1)
    assert len(ds) == 5
    x, y = ds[0]
    assert x.shape == (2,)
    assert y.shape == (3,)
    assert jnp.all((0 <= x) & (x <= 1))
    assert jnp.all((0 <= y) & (y <= 1))


def test_seed_reproducibility():
    ds1 = DemoDataset(size=3, seed=42)
    ds2 = DemoDataset(size=3, seed=42)
    assert jnp.array_equal(ds1.x, ds2.x)
    assert jnp.array_equal(ds1.y, ds2.y)


def test_dataloader_batch():
    ds = DemoDataset(size=8, seed=0)
    loader = DataLoader(ds, batch_size=4)
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (4, 2)
    assert y_batch.shape == (4, 3)
