"""Tests for model save/load utilities."""

import tempfile
from distinguish.model import build_model
from distinguish.io import save_model, load_model


def test_model_save_and_load():
    """Test that model weights can be saved and loaded correctly."""
    model = build_model(pretrained=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/weights.pt"
        save_model(model, path)
        # Modify one weight to ensure load actually changes state
        for p in model.parameters():
            p.data.add_(1.0)
            break
        load_model(model, path)
        # After load first param should not contain +1.0
        for p in model.parameters():
            assert (p.data.abs() < 0.9).any() or (p.data.abs() > 1.1).any() is False
            break
