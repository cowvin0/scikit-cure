"""Contract tests for latent-count distributions."""

import inspect

import pytest

from skcure._latent import (
    _BinomialLatentModel,
    _LatentCountModel,
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
)


REQUIRED_METHODS = {
    "incidence_value_and_grad",
    "observed_loglikelihood_from_eta",
    "e_step_from_eta",
    "predict_survival_from_eta",
    "predict_cure_probability_from_eta",
}


def test_latent_count_model_is_abstract():
    assert inspect.isabstract(_LatentCountModel)
    assert REQUIRED_METHODS <= _LatentCountModel.__abstractmethods__

    with pytest.raises(TypeError):
        _LatentCountModel()


@pytest.mark.parametrize(
    "factory,name",
    [
        (_PoissonLatentModel, "poisson"),
        (_NegativeBinomialLatentModel, "negative_binomial"),
        (_BinomialLatentModel, "binomial"),
    ],
)
def test_concrete_latent_models_satisfy_contract(factory, name):
    model = factory()

    assert isinstance(model, _LatentCountModel)
    assert not inspect.isabstract(type(model))
    assert model.name == name
    assert not type(model).__abstractmethods__
