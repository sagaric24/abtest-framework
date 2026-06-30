"""
bayesian.py — Bayesian A/B testing.

Covers:
  - Beta-Binomial model for conversion rates (conjugate, exact)
  - Normal-Normal model for continuous metrics
  - Probability that treatment > control
  - Expected loss calculation
  - Credible intervals

Bayesian testing advantages over frequentist:
  - No fixed sample size required
  - Results interpretable at any point (no peeking problem)
  - "Probability treatment is better" is intuitive for stakeholders
  - Expected loss quantifies business risk of choosing wrong variant
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


N_SAMPLES = 100_000  # Monte Carlo samples (increase for higher precision)


@dataclass
class BayesianResult:
    model: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift_pct: float

    prob_treatment_better: float        # P(treatment > control)
    prob_control_better: float          # P(control > treatment)
    expected_loss_treatment: float      # E[loss if we choose treatment]
    expected_loss_control: float        # E[loss if we keep control]

    credible_interval_lower: float      # CI for (treatment - control)
    credible_interval_upper: float
    ci_level: float

    decision: str
    loss_threshold: float

    control_n: int = 0
    treatment_n: int = 0

    def __str__(self) -> str:
        lines = [
            "=" * 58,
            f"  Bayesian A/B Test — {self.model}",
            "=" * 58,
            f"  Control   : mean={self.control_mean:.4f}  n={self.control_n:,}",
            f"  Treatment : mean={self.treatment_mean:.4f}  n={self.treatment_n:,}",
            f"  Absolute lift     : {self.absolute_lift:+.4f}",
            f"  Relative lift     : {self.relative_lift_pct:+.2f}%",
            "  ──────────────────────────────────────────",
            f"  P(treatment > ctrl): {self.prob_treatment_better:.1%}",
            f"  P(control > trt)  : {self.prob_control_better:.1%}",
            f"  Expected loss (trt): {self.expected_loss_treatment:.4f}",
            f"  Expected loss (ctrl): {self.expected_loss_control:.4f}",
            f"  {int(self.ci_level*100)}% Credible interval : "
            f"[{self.credible_interval_lower:.4f}, {self.credible_interval_upper:.4f}]",
            "  ──────────────────────────────────────────",
            f"  Decision : {self.decision}",
            "=" * 58,
        ]
        return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def bayesian_proportion_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    loss_threshold: float = 0.001,
    ci_level: float = 0.95,
    n_samples: int = N_SAMPLES,
) -> BayesianResult:
    """
    Beta-Binomial Bayesian test for conversion rates.

    Uses conjugate Beta prior → Beta posterior (exact, no approximation).

    Parameters
    ----------
    control_conversions, control_n : int
        Successes and total for control.
    treatment_conversions, treatment_n : int
        Successes and total for treatment.
    prior_alpha, prior_beta : float
        Beta prior hyperparameters. Default Beta(1,1) = uniform prior.
        Use Beta(2,20) for ~10% baseline if you have strong prior knowledge.
    loss_threshold : float
        Acceptable expected loss to declare a winner (in rate units).
        Default 0.001 = 0.1 percentage points.
    ci_level : float
        Width of credible interval. Default 0.95.
    n_samples : int
        Monte Carlo draws for posterior.

    Returns
    -------
    BayesianResult

    Examples
    --------
    >>> result = bayesian_proportion_test(500, 5000, 560, 5000)
    >>> print(result)
    """
    # Posterior parameters (conjugate update: Beta(α + conversions, β + non-conversions))
    alpha_ctrl = prior_alpha + control_conversions
    beta_ctrl = prior_beta + (control_n - control_conversions)
    alpha_trt = prior_alpha + treatment_conversions
    beta_trt = prior_beta + (treatment_n - treatment_conversions)

    # Sample from posteriors
    rng = np.random.default_rng(42)
    samples_ctrl = rng.beta(alpha_ctrl, beta_ctrl, n_samples)
    samples_trt = rng.beta(alpha_trt, beta_trt, n_samples)

    return _compute_result(
        samples_ctrl=samples_ctrl,
        samples_trt=samples_trt,
        ctrl_mean=alpha_ctrl / (alpha_ctrl + beta_ctrl),
        trt_mean=alpha_trt / (alpha_trt + beta_trt),
        control_n=control_n,
        treatment_n=treatment_n,
        model="Beta-Binomial (Conversion Rate)",
        loss_threshold=loss_threshold,
        ci_level=ci_level,
    )


def bayesian_means_test(
    control: np.ndarray,
    treatment: np.ndarray,
    loss_threshold: Optional[float] = None,
    ci_level: float = 0.95,
    n_samples: int = N_SAMPLES,
) -> BayesianResult:
    """
    Normal-Normal Bayesian test for continuous metrics (e.g., revenue, time).

    Uses non-informative (flat) Normal prior. Posterior is approximated
    via the Normal distribution using observed sufficient statistics.

    Parameters
    ----------
    control, treatment : array-like
        Raw observations per group.
    loss_threshold : float, optional
        Acceptable expected loss to declare winner. Defaults to 1% of ctrl mean.
    ci_level : float
        Credible interval width.

    Examples
    --------
    >>> ctrl = np.random.normal(50, 20, 5000)
    >>> trt  = np.random.normal(53, 20, 5000)
    >>> result = bayesian_means_test(ctrl, trt)
    >>> print(result)
    """
    ctrl = np.asarray(control, dtype=float)
    trt = np.asarray(treatment, dtype=float)

    # Posterior: Normal(sample_mean, sample_se²)
    # Under non-informative prior this is just the sampling distribution
    rng = np.random.default_rng(42)
    post_ctrl = rng.normal(ctrl.mean(), ctrl.std(ddof=1) / np.sqrt(len(ctrl)), n_samples)
    post_trt = rng.normal(trt.mean(), trt.std(ddof=1) / np.sqrt(len(trt)), n_samples)

    threshold = loss_threshold if loss_threshold is not None else abs(ctrl.mean()) * 0.01

    return _compute_result(
        samples_ctrl=post_ctrl,
        samples_trt=post_trt,
        ctrl_mean=float(ctrl.mean()),
        trt_mean=float(trt.mean()),
        control_n=len(ctrl),
        treatment_n=len(trt),
        model="Normal-Normal (Continuous Metric)",
        loss_threshold=threshold,
        ci_level=ci_level,
    )


# ── Internals ──────────────────────────────────────────────────────────────────

def _compute_result(
    samples_ctrl: np.ndarray,
    samples_trt: np.ndarray,
    ctrl_mean: float,
    trt_mean: float,
    control_n: int,
    treatment_n: int,
    model: str,
    loss_threshold: float,
    ci_level: float,
) -> BayesianResult:
    diff_samples = samples_trt - samples_ctrl

    prob_trt_better = float((diff_samples > 0).mean())
    prob_ctrl_better = 1.0 - prob_trt_better

    # Expected loss: how much we lose in expectation if we pick the wrong arm
    # If we choose treatment: loss = max(0, ctrl - trt) averaged over posterior
    expected_loss_trt = float(np.maximum(0, -diff_samples).mean())
    # If we keep control: loss = max(0, trt - ctrl) averaged over posterior
    expected_loss_ctrl = float(np.maximum(0, diff_samples).mean())

    # Credible interval for the difference
    tail = (1 - ci_level) / 2
    ci_lower, ci_upper = float(np.quantile(diff_samples, tail)), float(
        np.quantile(diff_samples, 1 - tail)
    )

    abs_lift = trt_mean - ctrl_mean
    rel_lift = (abs_lift / ctrl_mean * 100) if ctrl_mean != 0 else 0.0

    # Decision rule: choose arm whose expected loss is below threshold
    if expected_loss_trt < loss_threshold:
        decision = f"✅ Ship Treatment (E[loss]={expected_loss_trt:.4f} < {loss_threshold})"
    elif expected_loss_ctrl < loss_threshold:
        decision = f"🔁 Keep Control (E[loss of ctrl]={expected_loss_ctrl:.4f} < {loss_threshold})"
    else:
        decision = (
            f"⏳ Inconclusive — continue collecting data. "
            f"P(B>A)={prob_trt_better:.1%}"
        )

    return BayesianResult(
        model=model,
        control_mean=round(ctrl_mean, 4),
        treatment_mean=round(trt_mean, 4),
        absolute_lift=round(abs_lift, 4),
        relative_lift_pct=round(rel_lift, 2),
        prob_treatment_better=round(prob_trt_better, 4),
        prob_control_better=round(prob_ctrl_better, 4),
        expected_loss_treatment=round(expected_loss_trt, 5),
        expected_loss_control=round(expected_loss_ctrl, 5),
        credible_interval_lower=round(ci_lower, 4),
        credible_interval_upper=round(ci_upper, 4),
        ci_level=ci_level,
        decision=decision,
        loss_threshold=loss_threshold,
        control_n=control_n,
        treatment_n=treatment_n,
    )
