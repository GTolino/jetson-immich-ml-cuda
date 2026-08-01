"""Run two operations on the GPU: one that needs only CUDA, one that needs cuDNN.

Requesting CUDAExecutionProvider explicitly is the point: ONNX Runtime then
raises if it cannot load, instead of quietly falling back to CPU the way it
does when it is left to choose for itself.

TWO ops, because they fail independently and the difference is diagnostic:

  Relu  -- elementwise. Exercises the CUDA execution provider and nothing else.
           Passes on an image with every cuDNN engine library deleted.
  Conv  -- convolution. The CUDA provider implements this through cuDNN, whose
           engine libraries are `dlopen`ed at the moment the op first RUNS --
           not at image load, not at session creation. This is the only check
           in the suite that reaches them.

  Relu passes + Conv fails  ->  CUDA is fine, cuDNN is incomplete.
  Both fail                 ->  the CUDA provider itself is not loading.

Run inside the image under test:
    docker run --rm --runtime nvidia -v "$PWD/scripts:/s:ro" <image> \
      /opt/venv/bin/python /s/_gpu_smoke.py
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

PROVIDER = "CUDAExecutionProvider"
failures = []


def run(name, graph, feed, expected):
    """Build a one-op model, force it onto CUDA, check the arithmetic."""
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # Newer IR versions than the runtime supports are rejected outright.
    model.ir_version = 10
    path = "/tmp/_smoke_%s.onnx" % graph.name
    onnx.save(model, path)

    print("\n--- %s" % name)
    try:
        session = ort.InferenceSession(path, providers=[PROVIDER])
    except Exception as exc:  # noqa: BLE001 -- any failure is a failure
        print("FAIL: session creation raised: %s: %s" % (type(exc).__name__, exc))
        failures.append(name)
        return

    # ⚠ THE LINE THIS WHOLE SCRIPT TURNS ON.
    #
    # Without it, onnxruntime's Python wrapper catches an execution-provider
    # failure during run(), REBUILDS the session on CPU, retries, and returns a
    # correct answer -- printing only "Falling back to ['CPUExecutionProvider']"
    # amid the noise. The result is right, the device is wrong, and every
    # assertion below still passes. Verified by deleting a cuDNN engine library:
    # this script reported "produced a correct result on the GPU" for a
    # convolution that ran on the CPU.
    #
    # disable_fallback() makes run() RAISE instead of silently retrying.
    session.disable_fallback()

    providers = session.get_providers()
    print("session providers:", providers)
    if PROVIDER not in providers:
        print("FAIL: session did not get CUDA -- it is running on CPU")
        failures.append(name)
        return

    # The op does not execute until this line. A library that is `dlopen`ed on
    # first use -- every cuDNN engine -- surfaces here and nowhere earlier.
    try:
        out = session.run(None, feed)[0]
    except Exception as exc:  # noqa: BLE001
        print("FAIL: inference raised: %s: %s" % (type(exc).__name__, exc))
        failures.append(name)
        return

    # Belt and braces: re-read the providers AFTER the run. Session creation is
    # simply the wrong moment to ask -- that is the same "checked too early"
    # error that lazy loading punishes everywhere else in this stack.
    providers_after = session.get_providers()
    if PROVIDER not in providers_after:
        print("FAIL: fell back mid-run -- providers are now %s" % providers_after)
        failures.append(name)
        return

    print("output:", out.ravel())
    # Correct arithmetic rules out a stub library (every symbol resolves, nothing
    # computes). It does NOT rule out a correct answer from the wrong device --
    # that is what the two provider checks above are for.
    if not np.allclose(out, expected):
        print("FAIL: expected %s, got %s" % (np.asarray(expected).ravel(), out.ravel()))
        failures.append(name)
        return

    print("OK: %s produced a correct result on the GPU" % name)


# --- 1. Relu: the CUDA provider only, no cuDNN ---------------------------
relu_graph = helper.make_graph(
    [helper.make_node("Relu", ["X"], ["Y"])],
    "relu",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
)
run(
    "Relu (CUDA)",
    relu_graph,
    {"X": np.array([[-1, 2, -3, 4]], dtype=np.float32)},
    [[0, 2, 0, 4]],
)

# --- 2. Conv: goes through cuDNN -----------------------------------------
# 4x4 input, 3x3 kernel of ones, no padding, stride 1 -> 2x2 output.
# Every input is 1.0, so every output must be the window sum: 9.0.
conv_graph = helper.make_graph(
    [helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3])],
    "conv",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])],
    initializer=[
        helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 3, 3], [1.0] * 9),
    ],
)
run(
    "Conv (cuDNN)",
    conv_graph,
    {"X": np.ones((1, 1, 4, 4), dtype=np.float32)},
    [[[[9, 9], [9, 9]]]],
)

print()
if failures:
    print("RESULT: FAILED -- %s" % ", ".join(failures))
    sys.exit(1)
print("RESULT: both ops ran on the GPU with correct results")
