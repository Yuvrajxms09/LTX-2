import pytest
import torch

from ltx_pipelines.avatar.metrics import execution_snapshot, tensor_snapshot
from ltx_pipelines.avatar.runner import _require_inference_runtime


def test_require_inference_runtime_rejects_autograd_execution() -> None:
    with pytest.raises(RuntimeError, match=r"torch\.inference_mode"):
        _require_inference_runtime()


def test_require_inference_runtime_accepts_inference_execution() -> None:
    with torch.inference_mode():
        _require_inference_runtime()


def test_runtime_diagnostics_identify_inference_tensors() -> None:
    with torch.inference_mode():
        tensor = torch.ones(2, 3)
        execution = execution_snapshot()
        metadata = tensor_snapshot(tensor)

    assert execution == {
        "grad_enabled": False,
        "inference_mode_enabled": True,
    }
    assert metadata["shape"] == [2, 3]
    assert metadata["requires_grad"] is False
    assert metadata["is_inference"] is True
