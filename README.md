# Clarity

A residual U-Net for photograph denoising, trained from scratch in PyTorch.
Drop a photo into the web app, compare before and after with a slider, and download the result.

- **No sign-up needed.** Guests can restore photos right away; their photos are never written to disk.
- **Optional accounts** keep a history of your last 30 restorations.
- **Honest quality scores.** Add the clean original and the app shows PSNR and SSIM before and after.
- **Fast on a normal CPU.** A typical photo restores in a second or two, with no GPU required.
- **Optional "Enhance" mode** adds a Real-ESRGAN super-resolution pass that sharpens and upscales 4x — see below.

## Enhance mode (super-resolution)

The U-Net is a *denoiser*: it removes grain but cannot add sharpness or detail. For genuinely crisper, higher-resolution output there is an optional **Enhance** engine that runs Real-ESRGAN (`weights/realesrgan_x4.pth`, bundled). It is heavy — tens of seconds per photo on a CPU — so it is **off by default** and must be turned on explicitly:

```bash
ENABLE_ENHANCE=1 python app.py
```

A second "Enhance (sharpen + upscale)" option then appears under *Advanced options*. Set `ENHANCE_MAX_IN` (default 768) to cap how large the input is processed; the output is 4x that. Enhance is a local/demo feature — it is deliberately disabled on the free hosting tiers, which cannot run it.

## Run it locally

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Anchalpreet-15/VAE.git
cd VAE
pip install -r requirements.txt
python app.py
```

Your browser opens at <http://localhost:7860>. The trained model (`weights/unet.pth`, 16 MB) is included, so there is nothing else to download.

## Deploy it for free (Render)

Render builds from the `render.yaml` blueprint in the repository root. Nothing to configure by hand.

1. Sign in at <https://dashboard.render.com> (free account; GitHub login works).
2. Open <https://dashboard.render.com/select-repo?type=blueprint>, pick **Anchalpreet-15/VAE**, and set the branch (`main` once you've merged, or the feature branch to preview it first).
3. Click **Apply**. Render reads `render.yaml`, builds the Docker image, and starts a free web service at `https://clarity-<hash>.onrender.com`. The first build takes 6–8 minutes.

Every later push to the deployed branch redeploys automatically.

What the free plan means in practice:

| | |
|---|---|
| Hardware | 0.1 CPU, 512 MB RAM. The blueprint caps MAX_SIDE to 960 px and uploads to 6 MB so the app fits. Expect 5–10 s per photo. |
| Sleeping | Free web services spin down after 15 minutes of inactivity. The next visit waits about 40 s for the cold start. Open the URL a minute before you show it. |
| Disk | Free plans have no persistent disk. Accounts and saved history reset on every restart; guest use is unaffected. |

## Deploy it on Hugging Face Spaces (needs PRO)

As of late 2026, Hugging Face hosts **Docker** and Gradio Spaces on the free `cpu-basic` tier only with a **PRO** subscription ($9/month); solely static Spaces stay free. This app needs a Python backend, so it cannot run as a static Space. If you have (or want) PRO, the included workflow and script still deploy it:

- **Push-to-deploy:** add `HF_TOKEN` (a write token from *Settings, Access Tokens*) and `HF_SPACE` (`your-username/clarity`) as repository secrets under *Settings, Secrets and variables, Actions*, then push to `main`. The `deploy` workflow uploads the app and Hugging Face builds the image at `https://<owner>-<name>.hf.space`.
- **From your laptop:** `pip install huggingface_hub`, `huggingface-cli login`, then `python scripts/deploy_to_huggingface.py YOUR_USERNAME/clarity` (add `--dry-run` to preview the file list).

With PRO the hardware is 2 CPU cores and 16 GB RAM, and a Space sleeps only after a couple of idle days. Render (above) is the recommended free option.

## Run it anywhere with Docker

```bash
docker build -t clarity .
docker run -p 7860:7860 clarity
```

Then open <http://localhost:7860>. The same image runs on any host that takes a Dockerfile or a container (Fly.io, Railway, Google Cloud Run, a VPS).

## Results

The model was trained in two stages: first on artificially damaged BSDS500 photos (v1), then refined on realistic degradations and real camera-noise photos (v2, the version in this repository). Every number comes from `scripts/evaluate.py`, on photos the model never saw during training, with identical fixed random seeds for both versions.

**Real camera noise**: 36 crops from 10 scenes of the PolyU dataset that were held out from training entirely.

| | PSNR | SSIM |
|---|---|---|
| Noisy photo | 35.42 dB | – |
| v1 (first model) | 32.32 dB (-3.10) | 0.852 |
| **v2 (current)** | **36.54 dB (+1.12)** | **0.953** |

The first model had only ever seen heavy artificial noise, so it made real photos *worse*. The current one improves them. The typical gain is about +1 dB because real camera noise is mild; some photos gain 3 dB or more. The "Real camera photo" example in the app is one of the better cases.

**Synthetic damage**: 68 held-out BSDS500 photos. "Input" is the PSNR of the damaged photo; v1 and v2 show the gain in dB.

| Damage | Input | v1 | v2 |
|---|---|---|---|
| Barely visible noise (sigma 3) | 38.48 | -3.65 | **+0.88** |
| Noise sigma 10 | 28.25 | +4.63 | **+5.66** |
| Noise sigma 25 | 20.51 | +8.24 | **+8.58** |
| Noise sigma 50 | 14.97 | **+9.92** | +9.80 |
| Camera-style noise (signal-dependent) | 27.92 | +4.74 | **+5.70** |
| JPEG quality 35 | 30.29 | -0.86 | **+0.23** |
| Blur sigma 1.5 plus noise 20 | 20.54 | +4.45 | **+4.68** |
| Blur sigma 2.5 plus noise 10 | 22.32 | +1.80 | **+2.27** |
| Blur only, sigma 1 | 28.27 | -0.02 | -0.13 |
| Blur only, sigma 2 | 24.71 | **+1.09** | +1.00 |
| Blur only, sigma 3 | 23.28 | +0.76 | **+1.25** |
| Motion blur, 9 px | 24.16 | +0.04 | **+0.22** |
| Defocus blur, radius 3 | 24.94 | **+0.83** | +0.79 |
| Original mixed test set | 20.62 | 26.69 (+6.07) | **26.87 (+6.26)** |

## Limitations

- **Blur-only photos barely improve (about +1 dB).** The second training stage did not change this: the model is a denoiser first. Strong deblurring would need a larger model or a dedicated deblurring network trained for much longer, ideally on a GPU.
- **Real-noise results come from five camera models** (Canon 5D Mark II, 80D and 600D, Nikon D800, Sony A7 II) in one dataset, measured on 36 crops from 10 scenes. Phones and other cameras are untested.
- It cannot recover detail that is completely lost; very heavy noise leaves a soft finish.
- Photos larger than 1600 px on the long side are scaled down before processing to keep response times short. Raise `MAX_SIDE` (e.g. `MAX_SIDE=2560 python app.py`) for crisper full-resolution output on a machine with spare CPU and RAM; the free Render tier keeps it at 960.
- Only about 400 clean photos were available in the development environment (the large photo datasets were blocked there). Training on a bigger set such as DIV2K is the most promising way to improve quality further; `train.py --extra-clean DIR` adds extra photo folders.
- The real-noise photos and BSDS500 are licensed for non-commercial and research use, and the model was trained on them. Keep this project non-commercial (teaching, research, demonstrations).
- **NAFNet download URLs** are blocked from this environment and the two mirrors in `models/nafnet_loader.py` could not be verified. Loading code accepts official NAFNet-GoPro-width32.pth from https://github.com/megvii-research/NAFNet copied to `checkpoints/nafnet_weights.pth`.
- An optional pretrained **NAFNet** deblurring mode appears in the app when its weights are available (`python -m models.nafnet_loader` downloads them). Its download URLs were not reachable from the development environment, so that path is untested against the real weights; the architecture itself matches the official 17.11M-parameter model.

## Why is the repository called VAE when the model is a U-Net?

The repository is named `VAE` after the assignment prompt. The model that shipped is a deterministic residual U-Net. A probabilistic VAE was prototyped first (see the Git history) and shelved: at this dataset size a fixed point estimate of the clean photograph beats sampling from a learned latent prior, and the residual formulation avoids the characteristic VAE blur.

## How it works

The network is a U-Net with 3 down/up steps, skip connections, bilinear upsampling (no checkerboard artifacts) and a residual output, `tanh(f(x)) + x`, so it only has to learn the damage rather than redraw the photo. It has 7.8M parameters.

- **Stage 1 (v1):** 60 epochs on 360 BSDS500 photos, each with one fixed noisy or blurred copy, L1 loss.
- **Stage 2 (v2):** 8,000 further steps, about 75 minutes on 4 CPU cores. Photos are damaged freshly every time they are drawn (see `data/degradations.py`): untouched, noise, blur, blur plus noise, or JPEG only. Noise includes signal-dependent camera-style noise and strength is log-uniform so mild noise is common. Blur covers Gaussian, directional, motion and defocus kernels. 30% of each batch is real camera-noise patches from 25 PolyU scenes. The loss adds an edge term (a short A/B test showed it helps blur by about 0.2 dB and costs about 0.1 dB on real noise).

Efficiency work in this repository, measured on a 4-core CPU:

| Change | Effect |
|---|---|
| `channels_last` memory layout | inference 3.87 s to 3.30 s per 1024x768 photo; training step 19% faster |
| Automatic bfloat16 where the CPU benefits (decided by a timing test) | inference a further 3x faster (1.08 s) with test-set PSNR within 0.016 dB; training step 1.61 s to 0.48 s (3.4x) |
| Overlapping tile processing for large photos | peak extra memory for a 1280x960 photo drops from 2.8 GB to 0.8 GB; output matches whole-image processing to 64 dB |
| Half-precision weight file | 94 MB checkpoint becomes a 16 MB download with identical results |
| Damage generated on the fly | 1.4 ms per 128 px sample; every draw is a new noise realisation instead of one frozen copy |

On the server, one restoration runs at a time with a short queue, uploads are validated (type, size, pixel count, EXIF rotation, transparency), and sign-in and restore requests are rate limited per client.

## Train your own model

```bash
pip install -r requirements-train.txt
python data/prepare_data.py          # downloads BSDS500 and builds 360 train / 40 val / 68 test pairs
python train.py                      # from scratch, resumes automatically (add --epochs N for a shorter run)
```

Add real camera noise (optional, recommended):

```bash
git clone --depth 1 https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset
python scripts/prepare_real_data.py PolyU-Real-World-Noisy-Images-Dataset      # scene-level split
python train.py --init weights/unet.pth --epochs 16 --iters-per-epoch 500 --lr 2e-4 --grad-weight 0.5 --real-fraction 0.3
python scripts/evaluate.py --weights weights/unet.pth checkpoints/best_unet.pth --real data/real/test
python scripts/compare_models.py --weights weights/unet.pth checkpoints/best_unet.pth --names Before After --real data/real/test --out comparison.png
python scripts/export_weights.py     # optional: shrink best_unet.pth into weights/unet.pth
```

`python app.py` automatically prefers your own `checkpoints/best_unet.pth` over the shipped weights. Training uses bfloat16 automatically when your CPU benefits and resumes from `last_unet.pth`. No dataset access? `python data/prepare_data.py --synthetic` generates a small offline set to try the pipeline.

## Project structure

```
app.py                  web server: UI + JSON API (entry point)
config.py               hyperparameters and paths
train.py                training loop with resume
models/unet.py          residual U-Net
models/engine.py        inference: tiling, precision selection, weight loading
models/nafnet_loader.py optional pretrained NAFNet deblurring model
data/prepare_data.py    dataset download and fixed validation/test pairs
data/degradations.py    realistic noise, blur and JPEG damage applied on the fly
data/dataset.py         PyTorch datasets (on-the-fly, real patches, fixed pairs)
utils/                  metrics (PSNR/SSIM), image helpers, accounts and history storage
web/                    HTML, CSS and JavaScript for the interface, plus example photos
weights/unet.pth        trained model (16 MB)
scripts/                evaluate, compare_models, prepare_real_data, export_weights, deploy_to_huggingface
tests/                  pytest suite
Dockerfile              container image for deployment
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/restore` | multipart `image`, optional `reference`, `engine`; returns result URLs, timing and scores |
| `GET /api/result/{id}` | restored PNG (`?kind=input`, `?download=1`) |
| `POST /api/auth/signup`, `/login`, `/logout`; `GET /api/me` | optional accounts (bearer token) |
| `GET /api/history`, `/api/history/{id}/{input\|output\|thumb}`; `DELETE /api/history/{id}` | per-user history |
| `GET /api/status`, `/healthz` | available modes and limits; health check |

Settings are environment variables: `HOST`, `PORT`, `MAX_UPLOAD_MB`, `MAX_SIDE`, `MAX_QUEUE`, `DATA_DIR`, `RESTORE_PRECISION` (`auto`, `fp32`, `bf16`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the model, padding and tiling, metrics, the full API (guest and signed-in restores, history isolation between users, rate limits, history cap, bad uploads) and database recovery.

## Browser and platform compatibility

The UI is a plain HTML + CSS + JavaScript page with no framework and no bundler,
so it should work in any current browser. End-to-end flows were tested in
Chromium (it is what the development container had); Firefox and WebKit
were not verified here because their binaries could not be downloaded through
the proxy. Mobile layout was tested at 390 x 844 (iPhone-ish) with touch events.
The Docker image follows the usual pattern but was not actually built in the
development container (no Docker daemon); run `docker build -t clarity .` on
your machine before relying on it.

## Credits

Example photos and training data come from the Berkeley Segmentation Dataset (BSDS500): Martin, Fowlkes, Tal and Malik, *A Database of Human Segmented Natural Images*, ICCV 2001. Real camera-noise training data and the "Real camera photo" example come from the PolyU dataset: Xu, Li, Liang, Zhang and Zhang, *Real-world Noisy Image Denoising: A New Benchmark*, 2018 (license in `web/samples/NOTICE.txt`; non-commercial use). NAFNet: Chen, Chu, Zhang and Sun, *Simple Baselines for Image Restoration*, ECCV 2022.
