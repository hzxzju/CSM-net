from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from lazystrike_cls.classifier import (
    DinoV3LazyStrikeClassifier,
    LazyStrikePatchScorer,
    PseudoMaskPositionalAttention,
)


class TinyDinoV3(nn.Module):
    """Small DINOv3-compatible encoder for offline classification tests."""

    def __init__(self, hidden_size=32, patch_size=16, depth=4, registers=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, patch_size=patch_size)
        self.patch = nn.Conv2d(3, hidden_size, patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.registers = nn.Parameter(torch.zeros(1, registers, hidden_size))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU())
            for _ in range(depth)
        )

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        patch = self.patch(pixel_values).flatten(2).transpose(1, 2)
        batch = patch.shape[0]
        tokens = torch.cat(
            (self.cls.expand(batch, -1, -1), self.registers.expand(batch, -1, -1), patch),
            dim=1,
        )
        for block in self.blocks:
            tokens = tokens + block(tokens)
        return SimpleNamespace(last_hidden_state=tokens)


def make_classifier(**kwargs):
    return DinoV3LazyStrikeClassifier(
        encoder=TinyDinoV3(),
        encoder_dim=32,
        patch_size=16,
        num_classes=7,
        topk_ratio=0.25,
        **kwargs,
    )


def test_lazystrike_vote_invariant():
    scorer = LazyStrikePatchScorer(topk_ratio=0.25)
    tokens = torch.randn(2, 16, 8)
    cls_token = torch.randn(2, 8)
    result = scorer(tokens, cls_token)
    # Each of D channels casts K votes, hence sum_i(v_i) = D*K.
    assert result["vi"].shape == (2, 16)
    assert torch.all(result["vi"].sum(1) == 8 * 4)
    assert result["stability"].shape == tokens.shape
    assert result["vote_foreground_patch_mask"].dtype == torch.bool


def test_gaussian_filter_and_stability_match_official_repository_formula():
    scorer = LazyStrikePatchScorer(topk_ratio=0.25)
    tokens = torch.randn(2, 4, 8)
    cls_token = torch.randn(2, 8)

    result = scorer(tokens, cls_token)

    positions = torch.arange(-8 // 2 + 1, 8 // 2 + 1, dtype=tokens.dtype)
    expected_kernel = torch.exp(-0.5 * (positions / (8**0.5)).square())
    expected_kernel = expected_kernel / expected_kernel.max()
    assert torch.allclose(
        scorer._gaussian_filter(8, tokens.device, tokens.dtype),
        expected_kernel,
    )

    spectrum = torch.fft.fftshift(torch.fft.fft(tokens, dim=-1), dim=-1)
    expected_low_pass = torch.fft.ifft(
        torch.fft.ifftshift(spectrum * expected_kernel.view(1, 1, -1), dim=-1),
        dim=-1,
    ).real
    expected_stability = tokens / (expected_low_pass - tokens).abs().clamp_min(scorer.eps)
    assert torch.allclose(result["low_pass_tokens"], expected_low_pass)
    assert torch.allclose(result["stability"], expected_stability)


def test_patch_score_is_cosine_similarity_with_encoder_cls_token():
    scorer = LazyStrikePatchScorer(topk_ratio=0.5)
    tokens = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    cls_token = torch.tensor([[2.0, 0.0]])

    result = scorer(tokens, cls_token)

    expected = torch.tensor([[1.0, -1.0, 0.0, 2**-0.5]])
    assert torch.allclose(result["patch_score"], expected)


def test_pseudo_mask_positional_attention_uses_one_plus_gate():
    attention = PseudoMaskPositionalAttention(
        hidden_channels=4,
        foreground_prior=0.8,
    )
    feature = torch.arange(1, 25, dtype=torch.float32).reshape(1, 3, 2, 4)
    pseudo_mask = torch.tensor([[[1, 1, 0, 0], [1, 0, 1, 0]]], dtype=torch.bool)

    classification_feature, gate = attention(feature, pseudo_mask)

    assert gate.shape == (1, 1, 2, 4)
    assert torch.allclose(classification_feature, feature * (1.0 + gate))
    foreground = gate.squeeze(1).masked_select(pseudo_mask)
    background = gate.squeeze(1).masked_select(~pseudo_mask)
    assert torch.allclose(foreground, torch.full_like(foreground, 0.8))
    assert torch.allclose(background, torch.full_like(background, 0.2))


def test_end_to_end_classification_shapes_and_selected_mask():
    model = make_classifier().eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))

    assert output.logits.shape == (2, 7)
    assert output.probabilities.shape == (2, 7)
    assert output.prediction.shape == (2,)
    assert output.pooled_feature.shape == (2, 32)
    assert output.vi_map.shape == (2, 4, 4)
    assert output.patch_score_map.shape == (2, 4, 4)
    assert output.positional_gate.shape == (2, 1, 4, 4)
    assert output.positional_attention_map.shape == (2, 4, 4)
    assert output.pseudo_foreground_mask.shape == (2, 64, 64)
    assert torch.equal(output.pseudo_background_mask, ~output.pseudo_foreground_mask)
    assert torch.equal(
        output.selected_foreground_patch_mask,
        output.vote_foreground_patch_mask,
    )
    assert torch.allclose(output.probabilities.sum(-1), torch.ones(2), atol=1e-5)


def test_classification_loss_trains_attention_and_head_but_not_frozen_encoder():
    model = make_classifier().train()
    output = model(torch.randn(2, 3, 64, 64))
    loss = nn.functional.cross_entropy(output.logits, torch.tensor([1, 4]))
    loss.backward()

    assert model.classification_head[-1].weight.grad is not None
    assert model.positional_attention.refinement[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_rejects_non_patch_aligned_input():
    model = make_classifier()
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 63, 64))


def test_huggingface_dinov3_api_compatibility():
    transformers = pytest.importorskip("transformers")
    config = transformers.DINOv3ViTConfig(
        image_size=64,
        patch_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_register_tokens=4,
    )
    encoder = transformers.DINOv3ViTModel(config)
    model = DinoV3LazyStrikeClassifier(
        encoder=encoder,
        num_classes=7,
    ).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.logits.shape == (1, 7)
    assert output.vi_map.shape == (1, 4, 4)
