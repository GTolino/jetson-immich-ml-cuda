# Immich machine-learning with CUDA acceleration on Jetson (JetPack 6 / L4T r36.x).
#
# Upstream ships `immich-machine-learning:<ver>-cuda` for amd64 only, and the arm64
# image it does ship is CPU-only. This image merges three sources:
#
#   1. Immich's own arm64 image  -> the app, its uv-managed venv, Python 3.11
#   2. the onnxruntime builder   -> a Tegra CUDA wheel AND the CUDA/cuDNN userspace
#   3. the host, at runtime      -> libcuda.so.1, injected by the NVIDIA runtime
#
# Build the stage-1 image first; see docs/BUILDING-THE-WHEEL.md.

ARG IMMICH_VERSION=v3.0.3
ARG ORT_BUILDER=onnxruntime:1.24.1-builder-r36.5.tegra-aarch64-cp311-cu126-22.04

# ---------------------------------------------------------------- stage 1
FROM ${ORT_BUILDER} AS cuda

# ---------------------------------------------------------------- stage 2
FROM ghcr.io/immich-app/immich-machine-learning:${IMMICH_VERSION}

# --- CUDA userspace -----------------------------------------------------
# The container runtime injects only the DRIVER (libcuda.so.1) via drivers.csv.
# Everything below is userspace and must live in the image.
#
# The `.so.12*` / `.so.11*` globs are deliberate: they exclude the unversioned
# `libcublas.so`, which lives in `lib64/stubs/`. Stubs are EMPTY libraries used
# only to satisfy the linker at compile time -- ship one and it shadows the real
# library, every CUDA call resolves, and nothing computes.
COPY --from=cuda /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12*   /usr/local/cuda/lib64/
COPY --from=cuda /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.12*   /usr/local/cuda/lib64/
COPY --from=cuda /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublasLt.so.12* /usr/local/cuda/lib64/
COPY --from=cuda /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcufft.so.11*    /usr/local/cuda/lib64/

# cuDNN 9 is a ~130 KB dispatcher plus engine libraries it opens with dlopen at
# runtime -- `ldd` names only the dispatcher. Copy the dispatcher alone and the
# image passes every link check, then dies at the first convolution.
# `.so.9*` also skips `libcudnn_adv.so`, an absolute symlink into
# /etc/alternatives/ that would arrive dangling.
COPY --from=cuda /usr/lib/aarch64-linux-gnu/libcudnn*.so.9* /usr/local/cuda/lib64/

# Second path is where the NVIDIA runtime drops the host driver on JetPack 6.
# (JetPack 5 used /usr/lib/aarch64-linux-gnu/tegra -- adjust if you are on L4T 35.x.)
RUN printf '/usr/local/cuda/lib64\n/usr/lib/aarch64-linux-gnu/nvidia\n' \
      > /etc/ld.so.conf.d/cuda.conf \
 && ldconfig

# --- swap the CPU onnxruntime for the CUDA wheel -------------------------
# Both distributions install the SAME `site-packages/onnxruntime/` directory,
# so this is a replacement, not a coexistence. Installing over the top would
# leave orphaned CPU files and two .dist-info directories claiming ownership.
#
# Immich's venv is uv-managed and ships no pip; ensurepip unpacks the copy
# bundled in Python's stdlib. Always address the interpreter by full path --
# a login shell rebuilds PATH and loses /opt/venv/bin.
#
# The `&&` chain matters: create-and-delete inside ONE instruction keeps the
# wheel and pip cache out of every layer. Split across two RUNs, `rm` only
# writes a whiteout and you ship both anyway.
COPY --from=cuda /opt/onnxruntime_gpu-*.whl /tmp/
RUN /opt/venv/bin/python -m ensurepip \
 && /opt/venv/bin/python -m pip uninstall -y onnxruntime \
 && /opt/venv/bin/python -m pip install --no-deps /tmp/onnxruntime_gpu-*.whl \
 && rm -rf /tmp/*.whl /root/.cache/pip

# --- runtime switches ----------------------------------------------------
# NVIDIA_* are required: the stock Immich image does not set them, and without
# them the container runtime injects nothing at all -- silently.
#
# DEVICE is cosmetic for provider choice. Immich picks providers by filtering
# its SUPPORTED_PROVIDERS list against ort.get_available_providers(), with
# CUDA first and no env var involved. Set for parity with upstream's variants.
ENV DEVICE=cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia

# ENTRYPOINT and CMD are inherited (`tini -- python -m immich_ml`). Do not restate.
