"""
Phase 1 Risk Gate Verification: Automated ONNX Export & Runtime Tests
Validates that PointNet++ exports cleanly to ONNX opset 17 with dynamic point count,
loads in ONNX Runtime, matches PyTorch forward pass outputs, and relies exclusively
on standard ONNX Runtime C++ operators.
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.training.model import PointNet2SemSeg
from python.conversion.export_onnx import (
    export_model_to_onnx,
    validate_onnx_export,
    get_onnx_op_set,
    load_trained_model,
    STANDARD_ORT_CPP_OPS,
)


@pytest.fixture(scope="module")
def onnx_export_artifact():
    """
    Creates a temporary exported ONNX model from PointNet2SemSeg with random weights.
    Yields (model, onnx_filepath).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    model = PointNet2SemSeg(num_classes=3, in_channels=3).eval()
    temp_dir = tempfile.mkdtemp()
    onnx_path = os.path.join(temp_dir, "gate_test_pointnet2.onnx")

    export_model_to_onnx(
        model=model,
        output_path=onnx_path,
        nominal_n=4096,
        in_channels=3,
        opset_version=17,
        verbose=False,
    )
    yield model, onnx_path

    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)


def test_export_file_exists_and_valid_onnx(onnx_export_artifact):
    """Check 1: Export completes and creates a well-formed ONNX model."""
    _, onnx_path = onnx_export_artifact
    assert os.path.exists(onnx_path), "Exported ONNX file does not exist!"
    assert os.path.getsize(onnx_path) > 0, "Exported ONNX file is empty!"
    
    # onnx.checker validates structural integrity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    assert len(onnx_model.graph.node) > 0, "ONNX graph contains no nodes!"


def test_onnx_runtime_loads_model(onnx_export_artifact):
    """Check 2: ONNX Runtime loads the model successfully."""
    _, onnx_path = onnx_export_artifact
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    assert session is not None, "Failed to create ONNX Runtime InferenceSession!"
    
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    assert inputs[0].name == "points"
    assert outputs[0].name == "logits"


def test_output_shape_and_numerical_match_nominal(onnx_export_artifact):
    """Check 3 & 4: Output shape matches PyTorch exactly, and values match within tolerance (N=4096)."""
    model, onnx_path = onnx_export_artifact
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    N = 4096
    x_pt = torch.randn(1, N, 3, dtype=torch.float32)
    with torch.no_grad():
        out_pt = model(x_pt).numpy()

    ort_inputs = {"points": x_pt.numpy()}
    ort_out = session.run(["logits"], ort_inputs)[0]

    # Verify exact shape match
    assert ort_out.shape == out_pt.shape, f"Shape mismatch: {ort_out.shape} vs {out_pt.shape}"
    assert ort_out.shape == (1, N, 3)

    # Verify numerical closeness
    max_diff = np.max(np.abs(ort_out - out_pt))
    print(f"Max absolute difference at N={N}: {max_diff}")
    assert np.allclose(ort_out, out_pt, atol=1e-3, rtol=1e-3), (
        f"ONNX Runtime outputs deviate from PyTorch! Max diff: {max_diff}"
    )


@pytest.mark.parametrize("num_points", [2048, 6000])
def test_dynamic_axis_different_point_counts(onnx_export_artifact, num_points):
    """Check 5: Dynamic point count axis works for smaller (2048) and larger (6000) inputs."""
    model, onnx_path = onnx_export_artifact
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    x_pt = torch.randn(1, num_points, 3, dtype=torch.float32)
    with torch.no_grad():
        out_pt = model(x_pt).numpy()

    ort_inputs = {"points": x_pt.numpy()}
    ort_out = session.run(["logits"], ort_inputs)[0]

    assert ort_out.shape == (1, num_points, 3), (
        f"Dynamic shape mismatch: expected (1, {num_points}, 3), got {ort_out.shape}"
    )
    assert ort_out.shape == out_pt.shape

    max_diff = np.max(np.abs(ort_out - out_pt))
    assert np.allclose(ort_out, out_pt, atol=1e-3, rtol=1e-3), (
        f"ONNX numerical deviation at N={num_points}! Max diff: {max_diff}"
    )


def test_standard_onnx_runtime_cpp_operators_only(onnx_export_artifact):
    """Check 6: Op list contains nothing outside standard ONNX Runtime C++ support."""
    _, onnx_path = onnx_export_artifact
    ops = get_onnx_op_set(onnx_path)

    non_standard = [op for op in ops if op not in STANDARD_ORT_CPP_OPS]
    assert len(non_standard) == 0, (
        f"Found operators not supported by standard ONNX Runtime C++: {non_standard}"
    )


def test_missing_checkpoint_refuses_random_weight_export(tmp_path):
    """The CLI gate must fail loudly rather than export random weights."""
    missing_checkpoint = tmp_path / "missing-checkpoint.pth"
    with pytest.raises(FileNotFoundError, match="Refusing to export randomly initialized"):
        load_trained_model(str(missing_checkpoint))
