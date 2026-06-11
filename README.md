# 🔥 PyTorch Compatibility Matrix

**Live site: [torch-compat.danielhou.me](https://torch-compat.danielhou.me)**

Find compatible combinations of **PyTorch × Python × accelerator stack × OS**, the
minimum driver you need, and the exact `pip install` command — for every official
PyTorch build ever published:

- **NVIDIA CUDA** (cu75 → cu132) with minimum driver versions from the NVIDIA release notes
- **AMD ROCm** (3.7 → 7.x) with officially supported GPU architectures (gfx targets)
- **Intel XPU**
- **CPU / Apple Silicon (MPS)**
- Linux (x86_64, aarch64, s390x), Windows (x86_64, arm64), macOS (x86_64, arm64)

A GPU picker tells you which builds your card can run (by compute capability for
NVIDIA, gfx architecture for AMD).

## How it works

Everything is a static site — no server, no database:

1. `scripts/update_data.py` scrapes the official
   [PyTorch wheel index](https://download.pytorch.org/whl/torch/). Every wheel
   filename encodes a full compatibility record
   (`torch-2.5.1+rocm6.2.4-cp312-cp312-linux_x86_64.whl`), so one HTTP request
   yields the whole matrix → `data/torch_matrix.json`.
2. Hand-curated reference data lives in `data/`:
   `cuda_drivers.json` (CUDA → minimum driver), `cuda_sm_support.json`
   (CUDA → compute capabilities), `rocm_support.json` (ROCm → gfx targets),
   `gpus.json` (GPU name → architecture).
3. `scripts/build_site.py` renders it all into a single self-contained page
   (`_site/`) with client-side filtering in vanilla JS.

### Automatic updates

A [GitHub Actions workflow](.github/workflows/update-and-deploy.yml) runs **daily**:
it re-scrapes the wheel index, commits the data when PyTorch publishes new wheels
(new releases, new CUDA/ROCm/XPU variants), runs the tests, and redeploys the site
to GitHub Pages. New PyTorch releases appear on the site within a day, with no
manual steps.

## Running locally

```bash
pip install -r requirements.txt
python -m pytest                  # run tests
python scripts/update_data.py     # refresh data/torch_matrix.json
python scripts/build_site.py      # build into _site/
python -m http.server -d _site    # browse at http://localhost:8000
```

## Deploying your own

1. Push to GitHub and enable **Settings → Pages → Source: GitHub Actions**.
2. (Custom domain) Add a DNS `CNAME` record pointing your subdomain at
   `<username>.github.io`, and set the domain in **Settings → Pages**. The build
   already ships a `CNAME` file for `torch-compat.danielhou.me` — change
   `SITE_DOMAIN` in `scripts/build_site.py` for your own domain.
3. The scheduled workflow needs no secrets — it commits with the built-in
   `GITHUB_TOKEN`.

## Updating the curated data

The wheel matrix updates itself. The small curated files change only when NVIDIA
or AMD ship something new — each file carries a `_source` URL; PRs welcome.

## Acknowledgements

This project builds on
[**elenacliu/pytorch_cuda_driver_compatibilities**](https://github.com/elenacliu/pytorch_cuda_driver_compatibilities) —
thank you [@elenacliu](https://github.com/elenacliu) for the original idea, the
CUDA-toolkit-to-driver mapping, and the version-selection UX that this site
recreates. That project in turn built on
[moi90/pytorch_compute_capabilities](https://github.com/moi90/pytorch_compute_capabilities)
by Simon-Martin Schröder.

Useful references:

- [NVIDIA CUDA Toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) (driver table)
- [AMD ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- [Matching SM architectures to NVIDIA cards](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
- [PyTorch "get started" matrix](https://pytorch.org/get-started/locally/)

## License

[MIT](LICENSE)
