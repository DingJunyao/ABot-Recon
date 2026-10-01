import pytest

import abot_recon.model as model_module


def test_auto_prefers_paged(monkeypatch):
    monkeypatch.setattr(model_module, "flashinfer_available", lambda: True)
    assert model_module.resolve_attention_backend("auto") == "paged"


def test_auto_falls_back_to_sdpa(monkeypatch):
    monkeypatch.setattr(model_module, "flashinfer_available", lambda: False)
    assert model_module.resolve_attention_backend("auto") == "sdpa"


def test_explicit_paged_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(model_module, "flashinfer_available", lambda: False)
    with pytest.raises(RuntimeError, match="FlashInfer"):
        model_module.resolve_attention_backend("paged")


def test_explicit_sdpa_ignores_flashinfer(monkeypatch):
    monkeypatch.setattr(model_module, "flashinfer_available", lambda: True)
    assert model_module.resolve_attention_backend("sdpa") == "sdpa"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_bfloat16_attention_falls_back_when_flash_backend_is_unavailable():
    import torch

    from abot_recon.modeling.pi3.models.layers.attention import FlashAttention

    attention = FlashAttention(dim=64, num_heads=4).cuda().eval()
    x = torch.randn(1, 17, 64, device="cuda")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = attention(x)

    assert output.shape == x.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
