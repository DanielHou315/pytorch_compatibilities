# 🔥 PyTorch Compatibility Matrix — CUDA, ROCm, Python & Driver Versions

**Live site: [torch-compat.danielhou.me](https://torch-compat.danielhou.me)**

Which CUDA version works with my PyTorch? Which NVIDIA driver do I need? Does my
AMD GPU run the ROCm wheels? This site answers those questions for **every
official PyTorch build ever published**. Pick a combination of
**PyTorch × Python × accelerator stack × OS** and get the minimum driver and the
exact `pip install` command:

- **NVIDIA CUDA** (cu75 → cu132) with minimum driver versions from the NVIDIA release notes
- **AMD ROCm** (3.7 → 7.x) with officially supported GPU architectures (gfx targets)
- **Intel XPU**
- **CPU / Apple Silicon (MPS)**
- Linux (x86_64, aarch64, s390x), Windows (x86_64, arm64), macOS (x86_64, arm64)

A GPU picker tells you which builds your card can run (by compute capability for
NVIDIA, gfx architecture for AMD), including platform constraints — e.g. NVIDIA
Jetson is Linux aarch64 only and needs
[NVIDIA's JetPack wheels](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
rather than the SBSA server wheels on download.pytorch.org.

## Contributing

Contributions are welcome! If you spot missing or wrong data (a new GPU, CUDA
toolkit, driver or ROCm version) or want to improve the site, please
[fork the repository](https://github.com/DanielHou315/torch-compat/fork), make
your changes on a branch, and open a pull request.

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
