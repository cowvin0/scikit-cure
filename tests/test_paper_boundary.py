import importlib.util
from pathlib import Path
import subprocess
import sys

import skcure
import skcure.ensemble
import skcure.linear_model


def test_public_namespaces_have_no_paper_estimators():
    for name in (
        "GGPoissonPrimal",
        "GGNBPrimal",
        "SplinePoissonModel",
        "SplineNegativeBinomialModel",
    ):
        for namespace in (skcure, skcure.ensemble, skcure.linear_model):
            assert not hasattr(namespace, name)
            assert name not in namespace.__all__


def test_removed_generic_models_namespace_is_absent():
    assert importlib.util.find_spec("skcure.models") is None


def test_linear_model_module_has_no_paper_estimators():
    from skcure.linear_model import _cure

    for name in (
        "GGPoissonPrimal",
        "GGNBPrimal",
        "SplinePoissonModel",
        "SplineNegativeBinomialModel",
        "_SplineGGModel",
    ):
        assert not hasattr(_cure, name)


def test_importing_library_does_not_load_paper_models():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import skcure; "
                "assert not any(name == 'ggkm_paper' or name.startswith('ggkm_paper.') "
                "for name in sys.modules)"
            ),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
