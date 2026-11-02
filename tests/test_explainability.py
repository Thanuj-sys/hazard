"""Software-only attribution regression tests. No real evaluation claims."""
import pytest
import torch
from torch import nn
import evaluation.explainability as explanation


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([1., -1., .5]))
    def forward(self, inputs):
        image = inputs['images'] * inputs['quality_flags'][:, :, None, None, None]
        return image.flatten(1).sum(1, keepdim=True) * self.coefficients.view(1,-1)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(explanation, 'classification_logits', lambda model, inputs: model(inputs))
    model = LinearClassifier()
    inputs = {'images': torch.ones(1,1,2,2,2), 'quality_flags': torch.ones(1,1,dtype=torch.bool),
              'h3_indices': {}, 'geo_attributes': {}}
    return model, inputs


@pytest.mark.parametrize('method', ['gradient_times_input', 'integrated_gradients'])
def test_linear_attributions_match_known_contributions(setup, method):
    model,inputs = setup
    model.train()
    result = explanation.explain_image(model, inputs, method=method, steps=4)
    assert torch.allclose(result['attribution'], torch.ones_like(inputs['images']))
    assert result['metadata']['target_hazard'] == 'flood'
    assert result['metadata']['signed_attribution_sum'] == 8
    assert torch.allclose(result['channel_shares'],torch.tensor([.5,.5]))
    assert model.training
    assert model.coefficients.grad is None
    assert not inputs['images'].requires_grad
    if method == 'integrated_gradients':
        assert result['metadata']['completeness_residual'] == pytest.approx(0)


def test_explicit_class_target_preserves_signed_attribution(setup):
    model,inputs = setup
    result = explanation.explain_image(model,inputs,target=1)
    assert torch.all(result['attribution'] == -1)
    assert result['metadata']['target_hazard'] == 'wildfire'
    assert result['metadata']['predicted_hazard'] == 'flood'


def test_masked_frames_have_no_attribution(setup):
    model,inputs = setup
    inputs['images'] = torch.ones(1,2,2,2,2)
    inputs['quality_flags'] = torch.tensor([[True,False]])
    result = explanation.explain_image(model,inputs)
    assert not result['attribution'][0,1].any()
    assert torch.isfinite(result['heatmap']).all()


def test_zero_input_yields_explicit_zero_map(setup):
    model,inputs = setup
    inputs['images'].zero_()
    result = explanation.explain_image(model,inputs)
    assert result['metadata']['zero_attribution']
    assert result['heatmap'].sum() == 0
    assert result['channel_shares'].sum() == 0


def test_input_independent_classifier_yields_zero_map(setup,monkeypatch):
    model,inputs = setup
    monkeypatch.setattr(explanation,'classification_logits',lambda m,i: m.coefficients.view(1,3))
    result = explanation.explain_image(model,inputs)
    assert result['metadata']['zero_attribution']


def test_model_mode_restored_after_failure(setup):
    model,inputs = setup
    model.train()
    with pytest.raises(ValueError,match='target'):
        explanation.explain_image(model,inputs,target=99)
    assert model.training


def test_all_invalid_frames_rejected(setup):
    model,inputs = setup
    inputs['quality_flags'].zero_()
    with pytest.raises(ValueError,match='valid frame'):
        explanation.explain_image(model,inputs)


def test_nonfinite_image_rejected(setup):
    model,inputs = setup
    inputs['images'][0,0,0,0,0] = float('nan')
    with pytest.raises(ValueError,match='non-finite'):
        explanation.explain_image(model,inputs)


def test_multicase_batch_rejected(setup):
    model,inputs = setup
    inputs['images'] = torch.ones(2,1,2,2,2)
    with pytest.raises(ValueError,match='one case'):
        explanation.explain_image(model,inputs)
