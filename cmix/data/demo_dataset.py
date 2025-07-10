from dataclasses import dataclass

from jax import random
import jax.numpy as jnp


@dataclass
class DemoDataset:
    """Dataset mapping random 2-D points to RGB values via sinusoids."""

    size: int  # number of samples
    seed: int = 0  # RNG seed

    def __post_init__(self) -> None:
        key = random.PRNGKey(self.seed)
        key, x_key, amp_key = random.split(key, 3)
        self.x = random.uniform(x_key, (self.size, 2))
        self.amps = random.uniform(amp_key, (3, 2), minval=0.5, maxval=3.0)
        self.y = self._colors(self.x)

    def _colors(self, xs: jnp.ndarray) -> jnp.ndarray:
        x0, x1 = xs[:, 0], xs[:, 1]
        r = jnp.sin(self.amps[0, 0] * x0) + jnp.cos(self.amps[0, 1] * x1)
        g = jnp.sin(self.amps[1, 0] * x0) * jnp.cos(self.amps[1, 1] * x1)
        b = jnp.cos(self.amps[2, 0] * x0 + self.amps[2, 1] * x1)
        rgb = jnp.stack([(r + 2.0) / 4.0, (g + 1.0) / 2.0, (b + 1.0) / 2.0], axis=1)
        return rgb

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class DataLoader:
    """Minimal loader that batches a dataset."""

    def __init__(self, dataset: DemoDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            xs, ys = [], []
            for j in range(i, min(i + self.batch_size, len(self.dataset))):
                x, y = self.dataset[j]
                xs.append(x)
                ys.append(y)

            yield jnp.stack(xs), jnp.stack(ys)
