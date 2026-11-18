"""Realistic image degradations, applied on the fly while training.

Each sample is drawn from one scenario: untouched, noise only, blur only, blur plus noise, or
JPEG only (the first three also get a JPEG pass sometimes). The pieces are:

- blur: isotropic and anisotropic Gaussian, motion and defocus (disc) kernels.
- noise: per-channel and grey Gaussian, plus signal-dependent (Poisson-Gaussian)
  noise as produced by camera sensors, sometimes spatially correlated like the
  output of demosaicing. Strength is log-uniform, so mild noise is as common as
  heavy noise.
- JPEG re-compression at random quality.

A few samples are left untouched so the model learns to leave clean photos alone.
All functions take and return RGB images; ``random_degrade`` works on uint8 arrays.
"""

from __future__ import annotations

import cv2
import numpy as np

SCENARIOS = ("identity", "noise", "blur", "both", "jpeg")
SCENARIO_PROBABILITIES = (0.05, 0.35, 0.28, 0.27, 0.05)
P_JPEG_AFTER = 0.35


def gaussian_kernel(sigma_x: float, sigma_y: float, theta: float) -> np.ndarray:
    """Normalised (optionally rotated) anisotropic Gaussian blur kernel."""
    radius = int(np.ceil(3 * max(sigma_x, sigma_y)))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    u = np.cos(theta) * xx + np.sin(theta) * yy
    v = -np.sin(theta) * xx + np.cos(theta) * yy
    kernel = np.exp(-0.5 * ((u / sigma_x) ** 2 + (v / sigma_y) ** 2))
    return kernel / kernel.sum()


def motion_kernel(length: int, theta: float) -> np.ndarray:
    """Straight-line motion blur kernel of roughly ``length`` pixels."""
    size = length | 1
    centre = size // 2
    kernel = np.zeros((size, size), np.float32)
    dx, dy = np.cos(theta) * centre, np.sin(theta) * centre
    cv2.line(kernel, (round(centre - dx), round(centre - dy)), (round(centre + dx), round(centre + dy)), 1.0, 1)
    kernel = cv2.GaussianBlur(kernel, (0, 0), 0.6)
    return kernel / kernel.sum()


def disc_kernel(radius: float) -> np.ndarray:
    """Defocus (out-of-focus) blur kernel: a softly-edged disc."""
    size = int(np.ceil(radius)) * 2 + 3
    kernel = np.zeros((size, size), np.float32)
    cv2.circle(kernel, (size // 2, size // 2), int(round(radius)), 1.0, -1)
    kernel = cv2.GaussianBlur(kernel, (0, 0), 0.5)
    return kernel / kernel.sum()


def random_kernel(rng: np.random.Generator) -> np.ndarray:
    """Draw a blur kernel from the mix of blur types."""
    kind = rng.choice(4, p=[0.35, 0.25, 0.25, 0.15])
    if kind == 0:
        sigma = rng.uniform(0.5, 4.0)
        return gaussian_kernel(sigma, sigma, 0.0)
    if kind == 1:
        return gaussian_kernel(rng.uniform(0.5, 4.0), rng.uniform(0.5, 4.0), rng.uniform(0, np.pi))
    if kind == 2:
        return motion_kernel(int(rng.integers(5, 22)), rng.uniform(0, np.pi))
    return disc_kernel(rng.uniform(1.5, 5.0))


def add_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add sensor-like noise to a float32 image with values in [0, 255]."""
    x = img / 255.0
    kind = rng.choice(3, p=[0.4, 0.2, 0.4])
    if kind == 2:  # signal-dependent: variance = a * intensity + b
        a = np.exp(rng.uniform(np.log(2e-4), np.log(2e-2)))
        b = np.exp(rng.uniform(np.log(1e-6), np.log(2e-4)))
        noise = rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(np.clip(x, 0, 1) * a + b)
    else:
        sigma = np.exp(rng.uniform(np.log(1.0), np.log(50.0))) / 255.0
        if kind == 1:  # identical on every colour channel
            noise = np.repeat(rng.standard_normal(x.shape[:2] + (1,)).astype(np.float32), 3, axis=2) * sigma
        else:
            noise = rng.standard_normal(x.shape).astype(np.float32) * sigma
    if rng.random() < 0.3:  # spatially correlated, like demosaiced sensor noise
        smooth = cv2.GaussianBlur(noise, (0, 0), rng.uniform(0.4, 1.2))
        noise = smooth * (noise.std() / (smooth.std() + 1e-8))
    return (x + noise) * 255.0


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """Compress and decompress an RGB uint8 image as JPEG."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def random_degrade(clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a randomly degraded copy of an RGB uint8 image."""
    scenario = SCENARIOS[rng.choice(len(SCENARIOS), p=SCENARIO_PROBABILITIES)]
    if scenario == "identity":
        return clean.copy()
    img = clean.astype(np.float32)
    if scenario in ("blur", "both"):
        img = cv2.filter2D(img, -1, random_kernel(rng), borderType=cv2.BORDER_REFLECT)
    if scenario in ("noise", "both"):
        img = add_noise(img, rng)
    out = np.clip(img, 0, 255).round().astype(np.uint8)
    if scenario == "jpeg" or rng.random() < P_JPEG_AFTER:
        out = jpeg_roundtrip(out, rng.integers(30, 96))
    return out
