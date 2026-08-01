#!/usr/bin/env bash
# Verify a built image can actually reach the GPU -- before deploying it.
#
# Every check here exists because a weaker check passes while the GPU is unused.
# ONNX Runtime falls back to CPU silently, so "the container is healthy" and
# "the import worked" prove nothing at all.
#
# Usage: ./scripts/verify.sh [image-tag]

set -uo pipefail
IMAGE="${1:-immich-machine-learning:cuda-tegra-v3.0.3}"
VENV=/opt/venv/bin/python
CAPI=/opt/venv/lib/python3.11/site-packages/onnxruntime/capi
# Resolve from this script's own location, not the caller's cwd -- step 6 bind
# mounts this directory, and `./scripts/verify.sh` run from anywhere else would
# otherwise mount a path that does not exist.
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fail=0

echo "=== image: ${IMAGE}"

echo
echo "=== 1. interpreter (must be 3.11.x to match the wheel's ABI)"
docker run --rm "${IMAGE}" ${VENV} -V || fail=1

echo
echo "=== 2. installed distribution (expect exactly one onnxruntime/, one .dist-info)"
docker run --rm "${IMAGE}" bash -c \
  'ls /opt/venv/lib/python3.11/site-packages/ | grep -i "^onnxruntime"' || fail=1

echo
echo "=== 3. CUDA provider's link-time dependencies"
# NOTE: check the PROVIDER library, not onnxruntime_pybind11_state.so -- the
# latter is the Python binding layer and links no CUDA at all, so it reports
# clean on an image that has no CUDA whatsoever.
#
# The existence test is not redundant. If the library is absent, `ldd` errors,
# `grep` matches nothing, and the pipeline exits non-zero -- which would land in
# the `else` branch below and report "all resolved" for the worst possible
# image. Check the file is there before asking what it links against.
if ! docker run --rm "${IMAGE}" test -f "${CAPI}/libonnxruntime_providers_cuda.so"; then
  echo "  ABSENT -- libonnxruntime_providers_cuda.so is not in the image at all"
  echo "  (the wheel install did not take, or it was the CPU wheel)"
  fail=1
elif docker run --rm "${IMAGE}" bash -c \
     "ldd ${CAPI}/libonnxruntime_providers_cuda.so | grep 'not found'"; then
  echo "  ^^ MISSING -- copy these into the image"
  fail=1
else
  echo "  all resolved (link-time only -- says nothing about dlopen; see step 6)"
fi

echo
echo "=== 4. host driver reachable inside the container"
# Needs --runtime nvidia AND the NVIDIA_* env vars. Missing either injects
# nothing, with no error.
docker run --rm --runtime nvidia "${IMAGE}" ${VENV} -c \
  "import ctypes; print('cuInit:', ctypes.CDLL('libcuda.so.1').cuInit(0), '(0 = success)')" || fail=1

echo
echo "=== 5. providers COMPILED IN (necessary, NOT sufficient)"
# get_available_providers() reads a registry baked into the C++ binary at build
# time. It will happily list CUDAExecutionProvider on a machine with no CUDA
# libraries at all. It answers "what was I built with", not "what can I use".
docker run --rm --runtime nvidia "${IMAGE}" ${VENV} -c \
  "import onnxruntime as ort; print(ort.get_available_providers())" || fail=1

echo
echo "=== 6. real ops on real hardware  <-- the only checks that count"
# Requesting the provider EXPLICITLY makes ONNX Runtime raise instead of
# silently falling back, which is what turns this into a test.
#
# Two ops, deliberately. Relu exercises the CUDA provider alone; Conv goes
# through cuDNN, whose engine libraries are dlopen-ed at the moment the op
# first runs. Checks 1-5 and a Relu ALL PASS on an image with every cuDNN
# engine deleted -- the Conv is the only thing in this file that reaches them.
docker run --rm --runtime nvidia -v "${SCRIPTS}:/s:ro" "${IMAGE}" \
  ${VENV} /s/_gpu_smoke.py || fail=1

echo
if [ "${fail}" -ne 0 ]; then
  echo "RESULT: FAILED -- see above"
  exit 1
fi
cat <<'EOF'
RESULT: PASSED

One more check no script can do for you: run a real Smart Search job and watch
`tegrastats`. GR3D_FREQ must climb off 0%. A log line is the application's
opinion of itself; the hardware counter is not.
EOF
