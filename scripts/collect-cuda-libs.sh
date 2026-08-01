#!/usr/bin/env bash
# Collect the CUDA/cuDNN runtime libraries this image needs from the HOST's
# JetPack installation, into `cuda-libs.tar` for `Dockerfile.local`.
#
# Why this exists: the proven `Dockerfile` reads these libraries out of the
# jetson-containers builder image, which is ~26 GB and takes 5-9 hours to
# produce. But every JetPack 6 install already has the same files. Collecting
# them from the host means you only need the ~70 MB wheel, not the builder.
#
# Run this ON THE JETSON, from the repository root:
#     ./scripts/collect-cuda-libs.sh
#
# It writes `cuda-libs.tar` (a few hundred MB) into the build context.
# `.gitignore` excludes it -- it is NVIDIA's to distribute, not yours.

set -euo pipefail

CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/targets/aarch64-linux/lib}
SYS_LIB=${SYS_LIB:-/usr/lib/aarch64-linux-gnu}
OUT=${OUT:-cuda-libs.tar}
STAGE=$(mktemp -d)
trap 'rm -rf "${STAGE}"' EXIT

mkdir -p "${STAGE}/lib"

# Globs are deliberate and match the proven Dockerfile.
#
# `.so.12*` / `.so.11*` exclude the unversioned `libcublas.so`, which on a
# toolkit install points into `lib64/stubs/`. A stub is an EMPTY library that
# exists only to satisfy the linker at compile time. Ship one and it shadows the
# real library: every CUDA symbol resolves, and nothing computes.
#
# `.so.9*` likewise skips the unversioned `libcudnn_adv.so`, an absolute symlink
# into /etc/alternatives/ that would arrive in the image dangling.
PATTERNS=(
  "${CUDA_LIB}/libcudart.so.12*"
  "${CUDA_LIB}/libcublas.so.12*"
  "${CUDA_LIB}/libcublasLt.so.12*"
  "${CUDA_LIB}/libcufft.so.11*"
  "${SYS_LIB}/libcudnn*.so.9*"
)

missing=0
for pattern in "${PATTERNS[@]}"; do
  # shellcheck disable=SC2206
  matches=( ${pattern} )
  if [ ! -e "${matches[0]}" ]; then
    echo "MISSING: ${pattern}" >&2
    missing=1
    continue
  fi
  # -a preserves the symlinks (libcudnn.so.9 -> libcudnn.so.9.3.0). The loader
  # follows the SONAME, which is the versioned link, so both must be present.
  cp -a "${matches[@]}" "${STAGE}/lib/"
  printf '  %-28s %d file(s)\n' "$(basename "${pattern}")" "${#matches[@]}"
done

if [ "${missing}" -ne 0 ]; then
  cat >&2 <<'EOF'

Some libraries were not found on this host.

  libcudart / libcublas / libcufft   ->  sudo apt install cuda-libraries-12-6
  libcudnn*                          ->  sudo apt install libcudnn9-cuda-12

If your CUDA is not 12.6, set CUDA_LIB to the right targets/aarch64-linux/lib
path and adjust the version globs above to match.
EOF
  exit 1
fi

tar -C "${STAGE}" -cf "${OUT}" lib

echo
echo "wrote ${OUT}  ($(du -h "${OUT}" | cut -f1), $(tar -tf "${OUT}" | grep -c '\.so') libraries)"
echo
echo "Next: put the wheel in this directory, then"
echo "  docker build -f Dockerfile.local -t immich-machine-learning:cuda-tegra ."
