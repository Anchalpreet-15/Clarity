import numpy as np
import pytest

from data import degradations as dg
from data.dataset import MixedDataset, OnTheFlyDataset, RealPairDataset, RealPatchDataset
from train import RestorationLoss
from utils.metrics import psnr


@pytest.fixture()
def photo():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return np.ascontiguousarray(np.kron(base[::8, ::8], np.ones((8, 8, 1), dtype=np.uint8)))


@pytest.mark.parametrize("kernel", [
    dg.gaussian_kernel(2.0, 1.0, 0.7), dg.motion_kernel(11, 0.3), dg.disc_kernel(3.0),
])
def test_kernels_are_normalised(kernel):
    assert kernel.sum() == pytest.approx(1.0, abs=1e-5) and (kernel >= 0).all()


def test_degrade_keeps_shape_and_dtype(photo):
    rng = np.random.default_rng(1)
    for _ in range(40):
        out = dg.random_degrade(photo, rng)
        assert out.shape == photo.shape and out.dtype == np.uint8


def test_degrade_is_reproducible_and_covers_mild_and_heavy_damage(photo):
    first = [dg.random_degrade(photo, np.random.default_rng(i)) for i in range(5)]
    again = [dg.random_degrade(photo, np.random.default_rng(i)) for i in range(5)]
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    scores = np.array([psnr(dg.random_degrade(photo, np.random.default_rng(i)), photo) for i in range(300)])
    assert (scores >= 35).mean() > 0.08, "mild damage must be common so the model learns restraint"
    assert (scores < 22).mean() > 0.10, "heavy damage must be present too"
    assert (scores >= 98).any(), "some samples must be left untouched"


def test_on_the_fly_dataset_gives_fresh_degradations(tmp_path, photo):
    from PIL import Image

    Image.fromarray(photo).save(tmp_path / "a.png")
    ds = OnTheFlyDataset([tmp_path], patch_size=32, length=10)
    degraded, clean = ds[0]
    assert degraded.shape == clean.shape == (3, 32, 32) and 0.0 <= degraded.min() and degraded.max() <= 1.0
    assert len({float(ds[i][0].sum()) for i in range(6)}) > 1


def test_real_datasets_and_mixing(tmp_path, photo):
    from PIL import Image

    patches = np.zeros((4, 2, 48, 48, 3), np.uint8)
    patches[:, 1] = 200
    np.save(tmp_path / "p.npy", patches)
    real = RealPatchDataset(tmp_path / "p.npy", patch_size=32, length=8)
    noisy, clean = real[0]
    assert noisy.shape == (3, 32, 32) and float(clean.mean()) == pytest.approx(200 / 255, abs=1e-3)

    Image.fromarray(photo).save(tmp_path / "x_real.JPG")
    Image.fromarray(photo).save(tmp_path / "x_mean.JPG")
    assert len(RealPairDataset(tmp_path)) == 1

    mixed = MixedDataset(real, real, real_fraction=0.5, length=8)
    assert len(mixed) == 8 and mixed[3][0].shape == (3, 32, 32)


def test_edge_loss_prefers_sharp_results():
    target = torch_edges()
    blurry = torch_blur(target)
    loss = RestorationLoss(grad_weight=0.5)
    assert loss(target, target) < 1e-6 < loss(blurry, target)
    assert loss(blurry, target) > RestorationLoss(0.0)(blurry, target)


def torch_edges():
    import torch

    img = torch.zeros(1, 3, 16, 16)
    img[..., :, 8:] = 1.0
    return img


def torch_blur(img):
    import torch.nn.functional as F

    return F.avg_pool2d(F.pad(img, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
