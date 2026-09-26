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

DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "pointnet2_trained_best.pth"
)
DEFAULT_MODEL_METADATA_PATH = (
    REPO_ROOT / "models" / "metadata" / "model_metadata.json"
)
CANONICAL_CLASS_NAMES = ("terrain", "static_obstacle", "dynamic_object")

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


def load_trained_model(
    checkpoint_path: Optional[str],
    num_classes: int = 3,
    in_channels: int = 3,
    map_location: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any], Path]:
    """
    Load trained PointNet++ weights. Random initialization is never allowed here.

    Args:
        checkpoint_path: Explicit checkpoint path, or None to use the repository's
            trained checkpoint.
        num_classes: Expected number of output classes.
        in_channels: Expected number of input channels.
        map_location: Torch device used while loading the checkpoint.

    Returns:
        model: PointNet2SemSeg with checkpoint weights loaded.
        checkpoint: Raw checkpoint mapping, including stored evaluation metrics.
        resolved_path: Absolute checkpoint path actually used.

    Raises:
        FileNotFoundError: If no checkpoint exists at the requested location.
        ValueError: If the checkpoint is malformed or incompatible with the model.
    """
    candidate = (
        Path(checkpoint_path).expanduser()
        if checkpoint_path
        else DEFAULT_CHECKPOINT_PATH
    )
    resolved_path = candidate.resolve()
    if not resolved_path.is_file():
        requested = checkpoint_path if checkpoint_path else str(DEFAULT_CHECKPOINT_PATH)
        raise FileNotFoundError(
            "A trained PyTorch checkpoint is required for ONNX export. "
            f"Checkpoint not found: {requested}. "
            "Refusing to export randomly initialized weights."
        )

    try:
        checkpoint = torch.load(resolved_path, map_location=map_location)
    except Exception as exc:
        raise ValueError(
            f"Unable to load trained checkpoint: {resolved_path}: {exc}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint is not a checkpoint mapping: {resolved_path}. "
            "Refusing to export randomly initialized weights."
        )
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            f"Checkpoint contains no usable model_state_dict: {resolved_path}."
        )

    model = PointNet2SemSeg(num_classes=num_classes, in_channels=in_channels)
    try:
        load_result = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint weights are incompatible with "
            f"PointNet2SemSeg(num_classes={num_classes}, in_channels={in_channels}): "
            f"{resolved_path}: {exc}"
        ) from exc
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "Checkpoint did not load exactly. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}. "
            f"Checkpoint: {resolved_path}."
        )

    print(f"[Checkpoint] Loaded trained weights from {resolved_path}.")
    return model, checkpoint, resolved_path


def count_model_parameters(model: torch.nn.Module) -> int:
    """Return the exact number of parameters in a PyTorch module."""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def describe_onnx_contract(onnx_path: str) -> Dict[str, Any]:
    """
    Read the ONNX graph itself and report its deployable I/O contract.

    This prevents hand-written metadata from drifting away from tensor names,
    dynamic axes, channel count, class count, opset, or producer information.
    """
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    def shape_of(value_info) -> List[Any]:
        shape: List[Any] = []
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            return shape
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.dim_param:
                shape.append(str(dim.dim_param))
        return shape

    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError(
            f"Expected exactly one ONNX input and one ONNX output, found "
            f"{len(model.graph.input)} input(s) and {len(model.graph.output)} output(s): "
            f"{onnx_path}."
        )

    opset_versions = sorted(
        {opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")}
    )
    if not opset_versions:
        raise ValueError(f"ONNX model has no ai.onnx opset version: {onnx_path}.")

    return {
        "file_size_bytes": int(os.path.getsize(onnx_path)),
        "input_tensor_name": str(model.graph.input[0].name),
        "input_shape": shape_of(model.graph.input[0]),
        "output_tensor_name": str(model.graph.output[0].name),
        "output_shape": shape_of(model.graph.output[0]),
        "opset_version": int(opset_versions[-1]),
        "producer_name": str(model.producer_name or ""),
        "producer_version": str(model.producer_version or ""),
    }


def extract_checkpoint_metrics(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract stored evaluation metrics from a checkpoint mapping.

    The checkpoint observed in this repository stores overall accuracy, mean IoU,
    and per-class IoU/precision/recall. This function requires that evidence to
    be present instead of substituting plausible-looking numbers.
    """
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(
            "Checkpoint contains no stored evaluation metrics. "
            "Model metadata cannot be regenerated without rerunning evaluation."
        )

    required = ("overall_accuracy", "mean_iou", "per_class_iou")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(
            f"Checkpoint metrics are incomplete; missing {missing}. "
            "Model metadata cannot be regenerated without rerunning evaluation."
        )

    per_class_iou = metrics["per_class_iou"]
    if not isinstance(per_class_iou, dict):
        raise ValueError("Checkpoint per-class IoU is malformed.")
    missing_classes = [name for name in CANONICAL_CLASS_NAMES if name not in per_class_iou]
    if missing_classes:
        raise ValueError(
            f"Checkpoint metrics do not contain IoU for {missing_classes}."
        )

    return {
        "overall_accuracy": float(metrics["overall_accuracy"]),
        "mean_iou": float(metrics["mean_iou"]),
        "per_class_iou": {
            name: float(per_class_iou[name]) for name in CANONICAL_CLASS_NAMES
        },
        "per_class_precision": dict(metrics.get("per_class_precision", {})),
        "per_class_recall": dict(metrics.get("per_class_recall", {})),
        "best_miou": (
            float(checkpoint["best_miou"])
            if "best_miou" in checkpoint
            else float(metrics["mean_iou"])
        ),
    }


def save_model_metadata(
    output_metadata_path: str,
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    onnx_path: str,
    num_parameters: int,
) -> str:
    """
    Regenerate models/metadata/model_metadata.json from measured artifacts.

    Every reported number comes from the loaded checkpoint or the exported ONNX
    file. Nothing in this file is hand-copied from another report.
    """
    contract = describe_onnx_contract(onnx_path)
    metrics = extract_checkpoint_metrics(checkpoint)
    input_shape = contract["input_shape"]
    output_shape = contract["output_shape"]
    if len(input_shape) != 3 or len(output_shape) != 3:
        raise ValueError(
            f"Unexpected point-cloud tensor rank in {onnx_path}: "
            f"input={input_shape}, output={output_shape}."
        )

    metadata = {
        "model_name": "PointNet2_SemSeg",
        "task": "Semantic Segmentation for 2.5D Variable-Resolution LiDAR Mapping",
        "classes": {
            "0": CANONICAL_CLASS_NAMES[0],
            "1": CANONICAL_CLASS_NAMES[1],
            "2": CANONICAL_CLASS_NAMES[2],
        },
        "input_shape": input_shape,
        "input_tensor_name": contract["input_tensor_name"],
        "output_shape": output_shape,
        "output_tensor_name": contract["output_tensor_name"],
        "dynamic_axes": {"batch": 0, "num_points": 1},
        "num_parameters": int(num_parameters),
        "file_size_bytes": contract["file_size_bytes"],
        "accuracy_test_overall": metrics["overall_accuracy"] / 100.0,
        "mean_iou": metrics["mean_iou"] / 100.0,
        "class_iou": {
            name: metrics["per_class_iou"][name] / 100.0
            for name in CANONICAL_CLASS_NAMES
        },
        "checkpoint_metrics_percent": metrics,
        "deployment_target": "ONNX Runtime C++17",
        "training_framework": "PyTorch (Offline Only)",
        "training_torch_version": "not recorded in checkpoint",
        "export_producer": {
            "name": contract["producer_name"],
            "version": contract["producer_version"],
        },
        "export_environment": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "provenance": {
            "checkpoint_file": str(checkpoint_path),
            "onnx_file": str(Path(onnx_path).resolve()),
            "generator": "python/conversion/export_onnx.py:save_model_metadata",
            "regenerated_from_checkpoint_metrics": True,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_metadata_path)), exist_ok=True)
    with open(output_metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    print(f"[Metadata] Regenerated model metadata from measured artifacts: {output_metadata_path}")
    return output_metadata_path


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
    Validates an exported ONNX model against the supplied PyTorch module.

    The caller must supply a model whose trained weights have already loaded
    successfully. This function cannot detect random initialization on its own;
    the command-line entry point refuses to reach this validation step unless
    checkpoint loading succeeded exactly.
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

    Existing historical training provenance is preserved when present. The
    generator updates only the deployable contract and the checkpoint-derived
    training_info; it never deletes unrelated provenance fields.
    """
    if class_mapping is None:
        class_mapping = {
            "terrain": 0,
            "static_obstacle": 1,
            "dynamic_object": 2
        }

    existing_metadata: Dict[str, Any] = {}
    if os.path.exists(output_metadata_path):
        try:
            with open(output_metadata_path, "r", encoding="utf-8") as existing_file:
                loaded_metadata = json.load(existing_file)
            if isinstance(loaded_metadata, dict):
                existing_metadata = loaded_metadata
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Existing ONNX sidecar is unreadable, refusing to overwrite it: "
                f"{output_metadata_path}: {exc}"
            ) from exc

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
    if isinstance(existing_metadata.get("training_provenance"), dict):
        metadata["training_provenance"] = existing_metadata["training_provenance"]

    os.makedirs(os.path.dirname(os.path.abspath(output_metadata_path)), exist_ok=True)
    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"[Metadata] C++ runtime interface metadata saved to: {output_metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained PointNet++ to ONNX opset 17")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to trained PyTorch checkpoint (.pth). This is required.",
    )
    parser.add_argument("--output", type=str, default="models/onnx/pointnet2_semseg.onnx", help="Output .onnx path")
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(DEFAULT_MODEL_METADATA_PATH),
        help="Regenerated model-metadata JSON path.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--channels", type=int, default=3, help="Input channels (default: 3 for xyz)")
    parser.add_argument("--classes", type=int, default=3, help="Number of classes (default: 3)")
    args = parser.parse_args()

    try:
        model, checkpoint, checkpoint_path = load_trained_model(
            args.checkpoint,
            num_classes=args.classes,
            in_channels=args.channels,
        )
        checkpoint_metrics = extract_checkpoint_metrics(checkpoint)
        num_parameters = count_model_parameters(model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Checkpoint] ERROR: {exc}", file=sys.stderr)
        print(
            "[Checkpoint] Export aborted before tracing or validation; "
            "no random-weight ONNX artifact was created.",
            file=sys.stderr,
        )
        sys.exit(2)

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

    if not val_results["all_passed"]:
        print("\n==========================================")
        print(">>> ONNX RISK GATE VALIDATION: FAILED! <<<")
        print("Metadata was not regenerated because validation failed.")
        print("==========================================")
        sys.exit(1)

    meta_path = str(Path(export_path).with_suffix(".json"))
    save_onnx_metadata(
        meta_path,
        Path(export_path).name,
        opset_version=args.opset,
        in_channels=args.channels,
        num_classes=args.classes,
        training_info={
            "checkpoint_file": str(checkpoint_path),
            "checkpoint_metrics_percent": checkpoint_metrics,
            "num_parameters": num_parameters,
        },
    )
    save_model_metadata(
        args.metadata,
        checkpoint_path,
        checkpoint,
        export_path,
        num_parameters,
    )

    print("\n==========================================")
    print(">>> ONNX RISK GATE VALIDATION: PASSED! <<<")
    print("==========================================")
    sys.exit(0)
