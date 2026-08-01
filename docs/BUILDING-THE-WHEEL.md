# Building the `onnxruntime-gpu` wheel for Tegra

Only needed if you are **not** using a released wheel. Budget **5–9 hours** on an
Orin Nano and expect it to be memory-bound, not CPU-bound.

## Why it has to be built at all

Immich's ML service requires Python `>=3.11,<4.0`. The jetson-ai-lab package index
(`jp6/cu126`) ships `onnxruntime-gpu` as **`cp310` wheels only**.

A `cp310` wheel contains machine code compiled against CPython 3.10's internal C
API. It **physically cannot load** under 3.11 — no flag, no `--force`, no pip
option changes that. The only fix is recompiling from source.

That is the whole reason this repository exists. When a dependency has no build
for your platform, you patch the dependency, not the application.

## Prerequisites

- Jetson Orin (Nano / NX / AGX — all are `sm_87`) on **JetPack 6** (L4T r36.x)
- [`jetson-containers`](https://github.com/dusty-nv/jetson-containers), installed
- **~40 GB free disk** — the builder image alone lands around 26 GB
- **Swap.** 8 GB RAM is not enough on its own; see the memory section below

## The build

```bash
PYTHON_VERSION=3.11 CUDA_ARCHITECTURES=87 \
  jetson-containers build onnxruntime:1.24.1-builder
```

Run it **inside `tmux`**. A dropped SSH session takes the terminal with it, and a
build this long will outlive your connection:

```bash
tmux new -s ort
# ... start the build, then detach with Ctrl-b d
tmux attach -t ort
```

Three parameters, each load-bearing:

| Parameter | Why |
|---|---|
| `PYTHON_VERSION=3.11` | Must match the interpreter in Immich's image. Defaults to 3.12 on JetPack 6, which is wrong here. |
| `CUDA_ARCHITECTURES=87` | Orin's compute capability. Omit it and the build targets *every* known architecture, multiplying an already long build. |
| `:1.24.1-builder` | The `builder` variant sets `FORCE_BUILD=on`, compiling from source instead of pulling a prebuilt wheel. |

Verify the version you pick still satisfies Immich's constraint in
`machine-learning/pyproject.toml` at the release tag you are targeting.

## Memory

The compile runs *inside* a container, but a container is namespaces over one
kernel — **not** a VM. The build spends host RAM and swap and will OOM an 8 GB
Orin without help.

```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo sysctl vm.swappiness=10     # hoard RAM, swap reluctantly
```

`ccache` is enabled by default, so an OOM restart resumes rather than starting
over. Set `vm.swappiness` back to `60` afterwards — leaving it at 10 will hurt a
mixed workload, and the symptom (a sluggish resident LLM) looks nothing like the
cause.

> ⚠ An OOM here can be **silent**. If the kernel kills the process that was
> reporting progress, the build stops without an error message.

## Getting the wheel out

The build produces a large image whose only durable output is one file:

```bash
docker images | grep onnxruntime
# onnxruntime:1.24.1-builder-r36.5.tegra-aarch64-cp311-cu126-22.04   ~26 GB
```

That tag is worth reading — it encodes every dimension that had to be true at
once: L4T **r36.5**, **tegra** (not SBSA), **aarch64**, **cp311**, **cu126**.

```bash
docker run --rm <tag> ls -la /opt/*.whl
# /opt/onnxruntime_gpu-1.24.1-cp311-cp311-linux_aarch64.whl   (~70 MB)
```

Check the filename says `cp311` **and** `linux_aarch64` before going further.

Copy it to the host as insurance — 70 MB against a 9-hour rebuild:

```bash
mkdir -p ~/wheels
id=$(docker create <tag>)
docker cp $id:/opt/onnxruntime_gpu-1.24.1-cp311-cp311-linux_aarch64.whl ~/wheels/
docker rm $id
```

`docker create` builds a container's filesystem view without ever starting a
process — enough to read a file out of it.

## ⚠ Do not delete the builder image

`Dockerfile` reads **both** the wheel and the entire CUDA/cuDNN userspace from it
via `COPY --from=`. The host copy above only covers the wheel.

In particular, `docker image prune -a` removes every image not backing a
container — including this one. Your running ML container would survive; you
simply could never rebuild it. Plain `docker image prune` (no `-a`) is safe.
