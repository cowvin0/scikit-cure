"""Contracts for the shared, model-independent EM iteration driver."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from skcure._em import run_em


def _state():
    return SimpleNamespace(n_iter=0, converged=False, loglik_history=[])


def test_ordered_m_steps_receive_latest_parameters_and_one_shared_expectation():
    state = _state()
    calls = []

    def loglikelihood(parameters):
        calls.append(("likelihood", parameters))
        return float(parameters)

    def e_step(parameters):
        calls.append(("expectation", parameters))
        return {"expected": 3 * parameters}

    def first_m_step(parameters, expectation):
        calls.append(("first", parameters, expectation))
        return parameters + expectation["expected"]

    def second_m_step(parameters, expectation):
        calls.append(("second", parameters, expectation))
        return 2 * parameters

    with pytest.warns(ConvergenceWarning, match="EM did not converge within 2 iterations"):
        parameters, final_loglik = run_em(
            1,
            e_step=e_step,
            m_steps=(first_m_step, second_m_step),
            loglikelihood=loglikelihood,
            state=state,
            tol=0.0,
            max_iter=2,
        )

    assert parameters == 64
    assert final_loglik == 64.0
    assert state.loglik_history == [8.0, 64.0]
    assert state.n_iter == 2
    assert not state.converged
    assert [call[:2] for call in calls] == [
        ("likelihood", 1),
        ("expectation", 1),
        ("first", 1),
        ("second", 4),
        ("likelihood", 8),
        ("expectation", 8),
        ("first", 8),
        ("second", 32),
        ("likelihood", 64),
    ]
    assert calls[2][2] is calls[3][2]
    assert calls[6][2] is calls[7][2]
    assert calls[2][2] is not calls[6][2]


def test_normal_missing_data_em_can_use_dictionary_parameters():
    # One missing normal observation has conditional expectation equal to mu.
    # The observed-data likelihood is maximized at the observed mean, 3.
    observed = np.array([2.0, 4.0])
    state = _state()

    def update_mean(parameters, expected_missing):
        return {
            "mean": (observed.sum() + expected_missing) / 3.0,
            "updates": parameters["updates"] + 1,
        }

    parameters, final_loglik = run_em(
        {"mean": 0.0, "updates": 0},
        e_step=lambda parameters: parameters["mean"],
        m_steps=(update_mean,),
        loglikelihood=lambda parameters: -0.5 * np.sum(
            (observed - parameters["mean"]) ** 2
        ),
        state=state,
        tol=1e-12,
        max_iter=100,
    )

    assert parameters["mean"] == pytest.approx(3.0, abs=1e-6)
    assert parameters["updates"] == state.n_iter
    assert state.converged
    assert final_loglik == state.loglik_history[-1]
    assert final_loglik == pytest.approx(-1.0, abs=1e-12)
    assert np.all(np.diff(state.loglik_history) >= 0.0)


@pytest.mark.parametrize(
    "tol,max_iter,expected_history,converged",
    [
        (2.0, 1, [1.0], True),
        (1.0, 2, [1.0, 2.0], False),
        (0.0, 3, [1.0, 2.0, 3.0], False),
    ],
    ids=["converges-at-cap", "strict-tolerance", "honors-cap"],
)
def test_stopping_rules(tol, max_iter, expected_history, converged):
    state = _state()
    expectation = nullcontext() if converged else pytest.warns(
        ConvergenceWarning, match=f"EM did not converge within {max_iter} iterations"
    )
    with expectation:
        parameters, final_loglik = run_em(
            0,
            e_step=lambda parameters: None,
            m_steps=(lambda parameters, expectation: parameters + 1,),
            loglikelihood=float,
            state=state,
            tol=tol,
            max_iter=max_iter,
        )

    assert state.loglik_history == expected_history
    assert state.n_iter == len(expected_history)
    assert state.converged is converged
    assert parameters == len(expected_history)
    assert final_loglik == expected_history[-1]


def test_convergence_compares_against_the_previous_completed_likelihood():
    state = _state()
    parameters, final_loglik = run_em(
        0,
        e_step=lambda parameters: None,
        m_steps=(lambda parameters, expectation: parameters + 1,),
        loglikelihood=lambda parameters: float(min(parameters, 2)),
        state=state,
        tol=0.1,
        max_iter=10,
    )

    assert parameters == 3
    assert final_loglik == 2.0
    assert state.loglik_history == [1.0, 2.0, 2.0]
    assert state.n_iter == 3
    assert state.converged


@pytest.mark.parametrize("reject_initial", [False, True])
def test_initial_validator_runs_once_before_the_first_iteration(reject_initial):
    state = _state()
    calls = []

    def loglikelihood(parameters):
        calls.append(("likelihood", parameters))
        return float(parameters)

    def validate_initial(loglik):
        calls.append(("validate", loglik))
        if reject_initial:
            raise ValueError("Initial likelihood rejected")

    def e_step(parameters):
        calls.append(("expectation", parameters))

    arguments = dict(
        e_step=e_step,
        m_steps=(lambda parameters, expectation: parameters + 1,),
        loglikelihood=loglikelihood,
        state=state,
        tol=0.0,
        max_iter=2,
        validate_initial_loglik=validate_initial,
    )
    if reject_initial:
        with pytest.raises(ValueError, match="Initial likelihood rejected"):
            run_em(0, **arguments)
        assert calls == [("likelihood", 0), ("validate", 0.0)]
        assert state.n_iter == 0
        assert state.loglik_history == []
        assert not state.converged
    else:
        with pytest.warns(ConvergenceWarning, match="EM did not converge within 2 iterations"):
            run_em(0, **arguments)
        assert calls == [
            ("likelihood", 0),
            ("validate", 0.0),
            ("expectation", 0),
            ("likelihood", 1),
            ("expectation", 1),
            ("likelihood", 2),
        ]


def test_initial_likelihood_failure_precedes_validation_and_iteration():
    state = _state()

    def failing_likelihood(parameters):
        raise ArithmeticError("Cannot evaluate initial likelihood")

    def unexpected_call(*args):
        pytest.fail("No callback may run after initial likelihood failure")

    with pytest.raises(ArithmeticError, match="Cannot evaluate initial likelihood"):
        run_em(
            0,
            e_step=unexpected_call,
            m_steps=(unexpected_call,),
            loglikelihood=failing_likelihood,
            state=state,
            tol=0.1,
            max_iter=2,
            validate_initial_loglik=unexpected_call,
        )

    assert state.n_iter == 0
    assert state.loglik_history == []
    assert not state.converged


@pytest.mark.parametrize("bad_loglik", [np.nan, np.inf, -np.inf])
def test_nonfinite_update_keeps_only_completed_likelihood_history(bad_loglik):
    state = _state()

    with pytest.raises(FloatingPointError) as error:
        run_em(
            0,
            e_step=lambda parameters: None,
            m_steps=(lambda parameters, expectation: parameters + 1,),
            loglikelihood=lambda parameters: (
                float(parameters) if parameters < 2 else bad_loglik
            ),
            state=state,
            tol=0.0,
            max_iter=3,
            nonfinite_message="Toy likelihood failed.",
        )

    assert str(error.value) == "Toy likelihood failed."
    assert state.n_iter == 2
    assert state.loglik_history == [1.0]
    assert not state.converged


@pytest.mark.parametrize("failing_step", ["e", "m"])
def test_step_exception_preserves_progress_from_completed_iterations(failing_step):
    state = _state()

    def e_step(parameters):
        if failing_step == "e" and parameters == 1:
            raise RuntimeError("E-step failed")
        return None

    def m_step(parameters, expectation):
        if failing_step == "m" and parameters == 1:
            raise RuntimeError("M-step failed")
        return parameters + 1

    with pytest.raises(RuntimeError, match=f"{failing_step.upper()}-step failed"):
        run_em(
            0,
            e_step=e_step,
            m_steps=(m_step,),
            loglikelihood=float,
            state=state,
            tol=0.0,
            max_iter=3,
        )

    assert state.n_iter == 2
    assert state.loglik_history == [1.0]
    assert not state.converged
