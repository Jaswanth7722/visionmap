"""
ONNX Export and Validation Module for PointNet++ Semantic Segmentation
Exports PyTorch PointNet2SemSeg model to ONNX Opset 17 with dynamic point count axis.
Validates exported model against ONNX Runtime C++ standard operator set.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import onnx
import onnxruntime as ort

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.training.model import PointNet2SemSeg

# Standard ONNX Runtime C++ operators (excluding custom/contrib domains)
STANDARD_ORT_CPP_OPS = {
    "Abs", "Acos", "Acosh", "Add", "And", "ArgMax", "ArgMin", "Asin", "Asinh", "Atan", "Atanh",
    "AveragePool", "BatchNormalization", "BitShift", "BitwiseAnd", "BitwiseNot", "BitwiseOr", "BitwiseXor",
    "Cast", "CastLike", "Ceil", "Celu", "Clip", "Compress", "Concat", "ConcatFromSequence", "Constant",
    "ConstantOfShape", "Conv", "ConvInteger", "ConvTranspose", "Cos", "Cosh", "CumSum", "DepthToSpace",
    "DequantizeLinear", "Det", "Div", "Dropout", "Einsum", "Elu", "Equal", "Erf", "Exp", "Expand",
    "EyeLike", "Flatten", "Floor", "Gather", "GatherElements", "GatherND", "Gemm", "GlobalAveragePool",
    "GlobalLpPool", "GlobalMaxPool", "Greater", "GreaterOrEqual", "GridSample", "GroupNormalization",
    "HardSigmoid", "HardSwish", "Hardmax", "Identity", "If", "InstanceNormalization", "IsInf", "IsNaN",
    "LRN", "LSTM", "LeakyRelu", "Less", "LessOrEqual", "Log", "LogSoftmax", "Loop", "LpNormalization",
    "LpPool", "MatMul", "MatMulInteger", "Max", "MaxPool", "MaxRoiPool", "MaxUnpool", "Mean", "MeanVarianceNormalization",
    "Min", "Mod", "Mul", "Multinomial", "Neg", "NonMaxSuppression", "NonZero", "Not", "OneHot", "Or",
    "Pad", "Pow", "PRelu", "QLinearConv", "QLinearMatMul", "QuantizeLinear", "RandomNormal", "RandomNormalLike",
    "RandomUniform", "RandomUniformLike", "Range", "Reciprocal", "ReduceL1", "ReduceL2", "ReduceLogSum",
    "ReduceLogSumExp", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare",
    "Relu", "Reshape", "Resize", "ReverseSequence", "RoiAlign", "Round", "Scan", "Scatter", "ScatterElements",
    "ScatterND", "Selu", "SequenceAt", "SequenceConstruct", "SequenceEmpty", "SequenceErase", "SequenceInsert",
    "SequenceLength", "Shape", "Shrink", "Sigmoid", "Sign", "Sin", "Sinh", "Size", "Slice", "Softmax",
    "SoftmaxCrossEntropyLoss", "Softplus", "Softsign", "SpaceToDepth", "Split", "SplitToSequence", "Sqrt",
    "Squeeze", "StringNormalizer", "Sub", "Sum", "Tan", "Tanh", "TfIdfVectorizer", "ThresholdedRelu",
    "Tile", "TopK", "Transpose", "Trilu", "Unique", "Unsqueeze", "Upsample", "Where", "Xor"
}


def export_model_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    nominal_n: int = 4096,
    in_channels: int = 3,
    opset_version: int = 17,
    verbose: bool = True
) -> str:
    """
    Exports PointNet2SemSeg model to ONNX with dynamic point count axis.

    Args:
        model: PointNet2SemSeg PyTorch module
        output_path: Target .onnx file path
        nominal_n: Dummy input point count used for tracing
        in_channels: Number of input point cloud channels (default 3: x, y, z)
        opset_version: ONNX opset version (locked to 17 per specification)
        verbose: Whether to print export details
    Returns:
        output_path: Absolute path to exported model
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, nominal_n, in_channels, dtype=torch.float32)

    dynamic_axes = {
        "points": {0: "batch", 1: "num_points"},
        "logits": {0: "batch", 1: "num_points"}
    }

    if verbose:
        print(f"[ONNX Export] Tracing model with dummy input shape {dummy_input.shape}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["points"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False
    )

    if verbose:
        print(f"[ONNX Export] Successfully exported ONNX model to: {output_path}")

    return output_path


def get_onnx_op_set(onnx_path: str) -> List[str]:
    """Inspects an ONNX model and returns unique operator types."""
    model = onnx.load(onnx_path)
    ops = sorted(list(set(node.op_type for node in model.graph.node)))
    return ops


def validate_onnx_export(
    model: torch.nn.Module,
    onnx_path: str,
    test_counts: Tuple[int, ...] = (4096, 2048, 6000),
    in_channels: int = 3,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validates exported ONNX model with ONNX Runtime:
    - Verifies loading
    - Tests nominal point count
    - Tests varying point counts (dynamic axis verification)
    - Verifies output shape and numeric equivalence to PyTorch
    - Verifies all ops are supported standard ONNX Runtime C++ operators

    Returns:
        results: Dictionary containing test outcomes and op inspection
    """
    model.eval()

    # 1. Inspect operators
    ops = get_onnx_op_set(onnx_path)
    unsupported_ops = [op for op in ops if op not in STANDARD_ORT_CPP_OPS]

    if verbose:
        print(f"[ONNX Ops] Operators used in model ({len(ops)} total): {', '.join(ops)}")
        if unsupported_ops:
            print(f"[WARNING] Non-standard ops detected: {unsupported_ops}")
        else:
            print("[ONNX Ops] All operators are standard ONNX Runtime C++ supported operators!")

    # 2. Load with ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = {
        "onnx_path": onnx_path,
        "operators": ops,
        "unsupported_ops": unsupported_ops,
        "dynamic_axis_tests": []
    }

    # 3. Dynamic axis tests
    for N in test_counts:
        x_pt = torch.randn(1, N, in_channels, dtype=torch.float32)
        with torch.no_grad():
            out_pt = model(x_pt).numpy()

        x_np = x_pt.numpy()
        ort_out = session.run([output_name], {input_name: x_np})[0]

        shape_match = (ort_out.shape == out_pt.shape)
        max_diff = float(np.max(np.abs(ort_out - out_pt)))
        values_close = bool(np.allclose(ort_out, out_pt, atol=atol, rtol=rtol))

        test_result = {
            "num_points": N,
            "input_shape": list(x_np.shape),
            "expected_shape": list(out_pt.shape),
            "actual_shape": list(ort_out.shape),
            "shape_match": shape_match,
            "max_abs_diff": max_diff,
            "values_close": values_close
        }
        results["dynamic_axis_tests"].append(test_result)

        if verbose:
            status = "PASS" if (shape_match and values_close) else "FAIL"
            print(f"[Dynamic Test N={N}] Status: {status} | Shape: {ort_out.shape} | Max Diff: {max_diff:.2e}")

    all_passed = (
        len(unsupported_ops) == 0 and
        all(t["shape_match"] and t["values_close"] for t in results["dynamic_axis_tests"])
    )
    results["all_passed"] = all_passed
    return results


def save_onnx_metadata(
    output_metadata_path: str,
    onnx_filename: str,
    opset_version: int = 17,
    in_channels: int = 3,
    num_classes: int = 3,
    class_mapping: Optional[Dict[str, int]] = None,
    training_info: Optional[Dict[str, Any]] = None
):
    """
    Saves JSON metadata documenting the C++ runtime contract.
    """
    if class_mapping is None:
        class_mapping = {
            "terrain": 0,
            "static_obstacle": 1,
            "dynamic_object": 2
        }

    metadata = {
        "model_file": onnx_filename,
        "architecture": "PointNet++ (SSG Semantic Segmentation)",
        "opset_version": opset_version,
        "input_tensor": {
            "name": "points",
            "shape": ["batch", "num_points", in_channels],
            "data_type": "float32",
            "channel_order": ["x", "y", "z"] if in_channels == 3 else ["x", "y", "z", "intensity"],
            "memory_layout": "row-major flat array of N * channels floats (zero-copy from Point struct)"
        },
        "output_tensor": {
            "name": "logits",
            "shape": ["batch", "num_points", num_classes],
            "data_type": "float32",
            "interpretation": "Per-point raw logits across class dimension"
        },
        "classes": class_mapping,
        "c++_interface_notes": [
            "Input points tensor can have arbitrary point count N >= 512 (dynamic axis).",
            "Output logits can be mapped to predicted class via std::max_element / argmax along dim 2.",
            "No custom ONNX Runtime operator registration is required."
        ]
    }
    if training_info:
        metadata["training_info"] = training_info

    os.makedirs(os.path.dirname(os.path.abspath(output_metadata_path)), exist_ok=True)
    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Metadata] C++ runtime interface metadata saved to: {output_metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PointNet++ to ONNX opset 17")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="models/onnx/pointnet2_semseg.onnx", help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--channels", type=int, default=3, help="Input channels (default: 3 for xyz)")
    parser.add_argument("--classes", type=int, default=3, help="Number of classes (default: 3)")
    args = parser.parse_args()

    model = PointNet2SemSeg(num_classes=args.classes, in_channels=args.channels)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("Instantiating model with random weights (Phase 1 Risk Gate)...")

    export_path = export_model_to_onnx(
        model,
        args.output,
        nominal_n=4096,
        in_channels=args.channels,
        opset_version=args.opset
    )

    val_results = validate_onnx_export(
        model,
        export_path,
        test_counts=(4096, 2048, 6000),
        in_channels=args.channels
    )

    meta_path = str(Path(export_path).with_suffix(".json"))
    save_onnx_metadata(
        meta_path,
        Path(export_path).name,
        opset_version=args.opset,
        in_channels=args.channels,
        num_classes=args.classes
    )

    if val_results["all_passed"]:
        print("\n==========================================")
        print(">>> ONNX RISK GATE VALIDATION: PASSED! <<<")
        print("==========================================")
        sys.exit(0)
    else:
        print("\n==========================================")
        print(">>> ONNX RISK GATE VALIDATION: FAILED! <<<")
        print("==========================================")
        sys.exit(1)
