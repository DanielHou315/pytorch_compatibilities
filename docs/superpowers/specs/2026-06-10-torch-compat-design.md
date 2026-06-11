# torch-compat — PyTorch Compatibility Matrix Website

**Date:** 2026-06-10
**Deploy target:** GitHub Pages at `torch-compat.danielhou.me`
**Builds on:** [elenacliu/pytorch_cuda_driver_compatibilities](https://github.com/elenacliu/pytorch_cuda_driver_compatibilities)

> Note: designed and built in an autonomous session; decisions below were made
> without interactive review and are documented so they can be revisited.

## Goal

A static website that answers: *"Which PyTorch version / Python version /
accelerator stack (CUDA, ROCm, XPU, CPU) / OS combinations exist, and what
driver or GPU do I need?"* — kept up to date automatically by CI.

## Why not the original approach

The original repo scans the PyTorch **conda channel** (deprecated since
PyTorch 2.5) and runs `cuobjdump` on every package's `.so` files — heavy,
CUDA-toolkit-dependent, and frozen in time (last data ends at torch 2.3.1).
It also serves a **Streamlit** app, which cannot run on GitHub Pages.

## Architecture decision

**Data source:** the official PyTorch wheel index
(`https://download.pytorch.org/whl/torch/`). Every wheel filename encodes
`torch version + accelerator tag (cuXXX / rocmX.Y / xpu / cpu / none) +
Python ABI + platform`. One HTTP request yields the full matrix — verified
to cover cu92→cu132, rocm3.7→rocm7.2, xpu, and cpu. No wheel downloads, no
CUDA toolkit needed, runs in seconds in CI.

**Site:** Python static site generator (Jinja2) emits a single-page app:
embedded JSON dataset + vanilla JS filtering (dropdowns for PyTorch /
accelerator / Python / OS, like the original Streamlit UI). No JS build
toolchain. Hostable on GitHub Pages.

**Curated data** (small hand-maintained JSON files, updated rarely):
- `cuda_drivers.json` — CUDA toolkit → minimum NVIDIA driver (Linux & Windows), from NVIDIA release notes (extends the original repo's `cuda_driver.py`).
- `cuda_sm_support.json` — CUDA toolkit → supported compute capabilities (SM range).
- `rocm_support.json` — ROCm version → officially supported AMD GPU architectures (gfx targets).
- `gpus.json` — common GPU names → sm / gfx arch, for the GPU-lookup helper.

**Generated data:** `data/torch_matrix.json`, written by the scraper,
committed by CI so the repo always reflects the deployed site.

## Components

| Unit | Responsibility |
|---|---|
| `torchcompat/wheel_index.py` | Fetch + parse the wheel index into records (pure parsing functions, unit-tested) |
| `scripts/update_data.py` | CLI: scrape → aggregate → `data/torch_matrix.json` |
| `scripts/build_site.py` | CLI: join generated + curated data, render `templates/index.html.j2` + static assets → `_site/` |
| `static/app.js`, `static/style.css` | Client-side filtering UI |
| `.github/workflows/update-and-deploy.yml` | Daily cron + push + manual: scrape, commit data if changed, test, build, deploy to Pages |

## Data flow

wheel index → parse filenames → aggregate (torch, accel, accel_version,
python, os, arch) combos → join CUDA→driver and ROCm→gfx maps at build time →
embed as JSON in the page → JS filters client-side and renders the matching
rows + the right `pip install --index-url …` command.

## Error handling

- Scraper: non-200 or empty index → hard fail (CI keeps previous data; site still deploys from committed data).
- Unparseable wheel filename → skip + warn (never abort the whole scrape).
- Unknown CUDA/ROCm version in curated maps → row still rendered, driver column shows "?" (signal to update curated data).

## Testing

`pytest` over the pure functions: wheel filename parsing (incl. cp27mu-era
oddities, `none-macosx_*` MPS wheels, `+rocm6.2.4` multi-dot tags),
aggregation, and a build-output smoke test (HTML contains dataset + UI hooks).
CI runs tests before every deploy.

## Deployment

GitHub Actions → `actions/deploy-pages`, `CNAME` file with
`torch-compat.danielhou.me` included in the artifact. DNS: user adds a CNAME
record `torch-compat → danielhou315.github.io` (documented in README).
Schedule: daily at 06:17 UTC (cheap, catches releases within a day; PyTorch
stable releases are roughly bimonthly, ROCm/CUDA variant wheels appear
irregularly between them).
