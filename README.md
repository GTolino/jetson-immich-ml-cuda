# Immich machine learning with CUDA on a Jetson Orin

Immich doesn't publish a GPU machine-learning image for the Jetson. This is how I
built one, plus the tests that prove it's really using the GPU and not quietly
pretending.

**On an 8 GB Orin Nano:** Smart Search clears **100 assets in 3–4 seconds** —
roughly 1,700 a minute, against a CPU-only image slower by well over an order of
magnitude.

---

## Before you start

Verified on exactly one setup:

| | |
|---|---|
| Board | Jetson Orin Nano 8 GB |
| JetPack / L4T | JetPack 6 · **r36.5** |
| CUDA / cuDNN | **12.6** / **9.3.0** |
| Immich | **v3.0.3** |
| `onnxruntime-gpu` | **1.24.1**, built for **cp311** |

The other Orins (NX, AGX) are the same compute capability, `sm_87`, and should work
unchanged. **Xavier and the original Nano are not `sm_87`** and need their own build.

You'll also want Docker with the NVIDIA runtime (standard on JetPack), and —
if you're compiling the wheel yourself — about 40 GB free and some swap.

## Why there's no image to pull

The image bakes NVIDIA's cuDNN libraries on top of Immich, which is AGPL-3.0. The
[cuDNN SLA](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html)
§2.5 forbids putting the SDK under a licence that makes it "redistributable at no
charge" or "licensed for the purpose of making derivative works" — and AGPL-3.0
requires both.

Given that, I can't publish the resulting image. Below are the steps to build it
yourself, which takes about five minutes on your own box.

---

## Steps 1 & 2 — pick a route

The image needs two things: the **`onnxruntime-gpu` wheel** and the **CUDA/cuDNN
libraries**. There are two places to get them.

| | **Route A — the builder** | **Route B — download + host libs** |
|---|---|---|
| Wheel from | a builder image you compile | [Releases](../../releases) |
| CUDA libs from | that same builder image | your own JetPack install |
| Disk | ~40 GB | ~1 GB |
| Time | **5–9 hours** | **~5 minutes** |
| Status | ✅ proven in production | ✅ passes full verification |
| Dockerfile | [`Dockerfile`](Dockerfile) | [`Dockerfile.local`](Dockerfile.local) |

**Take Route B if your setup matches the table above.** Take Route A if it doesn't,
or if you need a different Immich or CUDA version.

Either way, run **Step 3** before you trust the result.

<details>
<summary><b>Why the wheel can't just come from an index</b></summary>

Immich's ML service needs Python 3.11+. The jetson-ai-lab index ships
`onnxruntime-gpu` for Tegra as **`cp310` wheels only**.

A `cp310` wheel holds machine code compiled against CPython 3.10's C API. It cannot
load under 3.11 — no flag, no `--force`, nothing. The only fix is recompiling.

</details>

### Route A — compile the wheel (5–9 hours)

Full details in **[docs/BUILDING-THE-WHEEL.md](docs/BUILDING-THE-WHEEL.md)**. Short
version:

```bash
PYTHON_VERSION=3.11 CUDA_ARCHITECTURES=87 \
  jetson-containers build onnxruntime:1.24.1-builder
```

Run it in `tmux`, give the box swap, and **don't delete the builder image afterwards**
— the Dockerfile pulls both the wheel and the CUDA libraries out of it.

```bash
docker build -t immich-machine-learning:cuda-tegra-v3.0.3 .
```

Override either input if your versions differ:

```bash
docker build \
  --build-arg IMMICH_VERSION=v3.0.3 \
  --build-arg ORT_BUILDER=onnxruntime:1.24.1-builder-r36.5.tegra-aarch64-cp311-cu126-22.04 \
  -t immich-machine-learning:cuda-tegra-v3.0.3 .
```

### Route B — download the wheel, use your own JetPack libraries

Your JetPack install already has the CUDA and cuDNN runtime libraries the builder
would have supplied. Grab them from the host:

```bash
./scripts/collect-cuda-libs.sh          # writes cuda-libs.tar
```

Then drop `onnxruntime_gpu-1.24.1-cp311-cp311-linux_aarch64.whl` from
[Releases](../../releases) into this directory and build:

```bash
docker build -f Dockerfile.local -t immich-machine-learning:cuda-tegra-v3.0.3 .
```

The wheel's filename is the compatibility contract — **`cp311`** and
**`linux_aarch64`** both have to be true for your box. If either isn't, you need
Route A.

Route B has been run end to end here and passes every check, including a convolution
through cuDNN, with results identical to Route A. What it hasn't had is a long
production run — Route A is the one that's served a real library for weeks. If Route B
gives you trouble, please open an issue.

Both Dockerfiles are commented line by line if you want to know why each `COPY` looks
the way it does.

## Step 3 — verify before deploying

```bash
./scripts/verify.sh immich-machine-learning:cuda-tegra-v3.0.3
```

Six checks, ending in two real ops on real hardware — one that needs only CUDA, one
that goes through cuDNN.

**Don't skip this, and don't substitute something lighter.** ONNX Runtime falls back
to CPU *silently* when a provider won't load: nothing errors, the container is
healthy, the API responds, jobs finish — slowly. Plenty of obvious-looking checks pass
on an image that never touches the GPU. [The appendix](#appendix-why-verification-is-not-optional)
names four of them.

What a pass looks like:

```
=== 6. real ops on real hardware  <-- the only checks that count

--- Relu (CUDA)
session providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
output: [0. 2. 0. 4.]
OK: Relu (CUDA) produced a correct result on the GPU

--- Conv (cuDNN)
session providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
output: [9. 9. 9. 9.]
OK: Conv (cuDNN) produced a correct result on the GPU

RESULT: PASSED
```

Both ops matter. Relu exercises the CUDA provider alone; Conv is the only thing here
that reaches cuDNN, whose engine libraries load on first use rather than at startup.
A suite without the Conv passes on an image with cuDNN deleted — not hypothetical,
that's how this one was caught being broken.

A `device_discovery.cc ... /sys/class/drm/card0/device/vendor` warning is normal on
Tegra. Ignore it.

## Step 4 — deploy

Adapt [`compose.yml`](compose.yml). The two essentials:

```yaml
image: immich-machine-learning:cuda-tegra-v3.0.3
runtime: nvidia
```

`compose.yml` assumes **remote ML** — Immich's server, database and web UI on another
host, reaching this box over the LAN on `:3003`. If you run everything on the Jetson,
lift the service block into that compose file in place of the upstream ML service.

Then start a Smart Search job and watch two independent things:

```bash
docker compose logs -f | grep -i "execution providers"   # want CUDAExecutionProvider first
tegrastats                                               # want GR3D_FREQ off 0%
```

A log line is the application's opinion of itself. The hardware counter isn't.

---

## Upgrading Immich

The upstream version lives in the Dockerfile's `FROM`, which is a **build-time** input.
Setting `IMMICH_VERSION` in a `.env` won't do it — a runtime variable can't select a
base image, because the base is resolved before any container exists.

Before bumping, check the **image**, not `pyproject.toml`:

```bash
docker run --rm ghcr.io/immich-app/immich-machine-learning:<new-version> \
  /opt/venv/bin/python -V
```

**Python version is the gate.** Still 3.11 → rebuild in minutes and re-run
`verify.sh`. Moved to 3.12 → the cp311 wheel is dead and it's another compile.

**Tag each build with its Immich version** — as above, `immich-machine-learning:cuda-tegra-v3.0.3`.
Then rolling back is just pointing `compose.yml` at the older tag and restarting; the
previous image is still there under its own name. Reuse one tag across builds and
`docker build -t` moves the name to the new image, leaving the old one as a dangling
`<none>` you can only recover by digest.

While we're here: this is also the durable answer to patching a container. **You
don't.** Express the patch as a Dockerfile and rebuild against new upstream. A
`docker exec` fix survives exactly until the next image update.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `verify.sh` step 3 lists `not found` libraries | A `COPY` glob missed a file. Route A: the builder image is a different CUDA version than the Dockerfile expects. Route B: `collect-cuda-libs.sh` didn't find everything — re-run it and read its output |
| `collect-cuda-libs.sh` reports MISSING | Your JetPack lacks those packages, or your CUDA isn't 12.6. The script prints the `apt` line for each; set `CUDA_LIB` if your toolkit path differs |
| Step 4 can't load `libcuda.so.1` | Missing `--runtime nvidia`, or the `NVIDIA_*` env vars are absent — either one injects nothing, with no error |
| `device_discovery.cc ... "/sys/class/drm/card0/device/vendor"` | **Harmless.** ONNX Runtime 1.24 enumerates GPUs via DRM sysfs nodes; Tegra's GPU isn't a discrete PCI device and has no `card0`. CUDA initialises fine right after it |
| Relu passes, Conv fails | CUDA works, **cuDNN doesn't**. An engine library is missing or the wrong version — that split is exactly why there are two ops |
| Step 5 passes, step 6 fails entirely | Providers were compiled in; their libraries aren't loadable |
| Everything passes, `GR3D_FREQ` stays 0% | Immich isn't calling this container. Check the server's ML URL points at this host's LAN address and port |
| Container healthy, jobs slow | Silent CPU fallback. Re-run `verify.sh` — a healthy container proves nothing about the GPU |

---

## Background

Skip this if it already works. It's here because each piece cost hours to find.

### How the pieces fit

Four layers have to be present, and each has a different owner:

| Layer | What it is | Where it comes from |
|---|---|---|
| **GPU driver** — `libcuda.so.1` | kernel/hardware interface | **the host**, injected at runtime by the NVIDIA container runtime |
| **CUDA userspace** — `libcudart`, `libcublas`, `libcudnn`, … | the math libraries | **baked into the image** — nothing injects these |
| **`onnxruntime-gpu`** | Python library with CUDA compiled in | the wheel from step 1 |
| **Immich ML app** | the service calling ONNX Runtime | Immich's own arm64 image |

Immich's arm64 image gives you the last layer and gets the driver free at runtime. It
has no CUDA userspace at all, and its ONNX Runtime is the CPU build. **This repo adds
the middle two.**

You can confirm the runtime really does inject only the driver:

```bash
ls /etc/nvidia-container-runtime/host-files-for-container.d/
# devices.csv  drivers.csv     <- no cuda.csv, no cudnn.csv
```

### `arm64` is not one platform

The trap that costs the most time. `nvidia/cuda:*-runtime-ubuntu22.04` has an arm64
build, but it's **SBSA** — server-class ARM with discrete GPUs. Jetson is **Tegra**:
the GPU is on the SoC and the CUDA driver ships with the L4T board support package.

The SBSA image pulls happily on a Jetson and simply doesn't work. Both are
`linux/arm64` and nothing in the tag warns you. That's why dusty-nv hand-encodes
`r36.5-tegra-aarch64-cp311-cu126-22.04` into his tags: **the standard platform labels
are coarser than real compatibility.**

### Appendix: why verification is not optional

Four ways to fool yourself. The first three turned up while building the image; the
fourth turned up while verifying it, and it defeated an earlier version of the script
in `scripts/`.

**`get_available_providers()` is not a capability check.** It returns a list compiled
into the C++ binary at build time. On a machine missing *every* CUDA library it still
reported `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`.
It answers "what was I built with", not "what can I use".

**`ldd` on the wrong file reports clean.** `onnxruntime_pybind11_state.so` is the
Python binding layer and links no CUDA at all. The CUDA dependencies live in
`libonnxruntime_providers_cuda.so`.

**`ldd` can't see `dlopen`.** cuDNN 9 is a **130 KB dispatcher** plus engine libraries
it loads on demand — `libcudnn_engines_precompiled.so.9` alone is **510 MB**. `ldd`
names only the dispatcher. Copy just that and the image passes every link check, then
dies at the first convolution.

**And a fallback will hide even that.** Found by deleting
`libcudnn_engines_precompiled.so.9*` from a working image and running the suite
against it. cuDNN failed exactly as expected:

```
Unable to load any of {libcudnn_engines_precompiled.so.9.3.0, ... }
cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
```

…and then ONNX Runtime's Python wrapper caught the failure, **rebuilt the session on
CPU, retried, and returned the right answer.** The smoke test printed "produced a
correct result on the GPU" and the suite exited `PASSED`. The provider list had been
read at session creation, before the fallback; the arithmetic was correct because the
CPU computed it correctly.

Two fixes, both one-liners: call **`session.disable_fallback()`** so a provider failure
raises instead of retrying elsewhere, and **re-read the providers after `run()`**, not
only before it. Correct output proves the math, not the device.

The shape of all four: **lazy loading trades a loud failure at startup for a quiet one
at runtime** — and a fallback turns that quiet failure into an apparent success.

---

## Known limitations

- **Memory on an 8 GB board is untested here.** If you also run a resident LLM, that's
  three workloads and a CUDA context each sharing one pool.
- **TensorRT isn't shipped.** The wheel builds
  `libonnxruntime_providers_tensorrt.so`, but its libraries are deliberately left out —
  Immich's `SUPPORTED_PROVIDERS` doesn't include TensorRT, so it's never requested.
- **Only Immich v3.0.3 has run in production.** It ships onnxruntime 1.26.0; this
  replaces it with 1.24.1, inside the declared `>=1.23.2,<2` range. Newer releases may
  use APIs 1.24.1 lacks.

## Credits

The heavy lifting is [`jetson-containers`](https://github.com/dusty-nv/jetson-containers)
by dusty-nv, which turns the wheel build into a single command. This repo is the
assembly step and the verification around it.

## License

Everything in this repository — Dockerfiles, scripts, docs — is **MIT**.

That covers the repository, not what a build of it produces:

- **NVIDIA's CUDA and cuDNN libraries** land in the image under the
  [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) and the
  [cuDNN SLA](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html).
- **Immich** is **AGPL-3.0**, and an image built from these instructions is a derived
  work of it.

Building it for yourself is fine — you accepted NVIDIA's terms when you installed
JetPack. Redistributing the built image isn't, which is why this is a tutorial rather
than a `docker pull`.
