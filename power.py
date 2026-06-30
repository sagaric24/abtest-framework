"""
power.py — Sample size & power calculations for A/B tests.

Supports:
  - Two-proportion z-test (conversion rates)
  - Two-sample t-test (continuous metrics like revenue)
  - One-sided and two-sided tests
  - MDE (minimum detectable effect) calculation
  - Power curves
"""

from __future__ import annotations
import math
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PowerResult:
    sample_size_per_group: int
    total_sample_size: int
    alpha: float
    power: float
    mde: float
    effect_size: float
    test_type: str
    alternative: str

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "  Power Analysis Result",
            "=" * 50,
            f"  Test type       : {self.test_type}",
            f"  Alternative     : {self.alternative}",
            f"  Alpha (α)       : {self.alpha}",
            f"  Power (1-β)     : {self.power}",
            f"  Effect size     : {self.effect_size:.4f}",
            f"  MDE             : {self.mde:.4f}",
            f"  Per-group n     : {self.sample_size_per_group:,}",
            f"  Total n         : {self.total_sample_size:,}",
            "=" * 50,
        ]
        return "\n".join(lines)


def sample_size_proportion(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> PowerResult:
    """
    Calculate required sample size for a proportion test (e.g., conversion rate).

    Parameters
    ----------
    baseline_rate : float
        Current conversion rate in control (e.g., 0.10 for 10%).
    mde : float
        Minimum detectable effect as absolute change (e.g., 0.02 for +2pp).
    alpha : float
        Type I error rate. Default 0.05.
    power : float
        Desired statistical power (1 - β). Default 0.80.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> result = sample_size_proportion(baseline_rate=0.10, mde=0.02)
    >>> print(result)
    """
    treatment_rate = baseline_rate + mde

    # Pooled proportion under H0
    p_pool = (baseline_rate + treatment_rate) / 2

    # Z-scores
    z_alpha = _z_alpha(alpha, alternative)
    z_beta = stats.norm.ppf(power)

    # Standard errors
    se_h0 = math.sqrt(2 * p_pool * (1 - p_pool))
    se_h1 = math.sqrt(
        baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)
    )

    # Cohen's h effect size
    h = 2 * math.asin(math.sqrt(treatment_rate)) - 2 * math.asin(
        math.sqrt(baseline_rate)
    )

    n = ((z_alpha * se_h0 + z_beta * se_h1) / abs(mde)) ** 2
    n = math.ceil(n)

    return PowerResult(
        sample_size_per_group=n,
        total_sample_size=n * 2,
        alpha=alpha,
        power=power,
        mde=mde,
        effect_size=abs(h),
        test_type="Two-proportion z-test",
        alternative=alternative,
    )


def sample_size_means(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> PowerResult:
    """
    Calculate required sample size for a means test (e.g., average revenue per user).

    Parameters
    ----------
    baseline_mean : float
        Mean in control group.
    baseline_std : float
        Standard deviation (assumed equal across groups).
    mde : float
        Minimum detectable effect (absolute difference in means).
    alpha : float
        Type I error rate.
    power : float
        Desired power.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> result = sample_size_means(baseline_mean=50.0, baseline_std=20.0, mde=5.0)
    >>> print(result)
    """
    z_alpha = _z_alpha(alpha, alternative)
    z_beta = stats.norm.ppf(power)

    # Cohen's d
    d = abs(mde) / baseline_std

    n = math.ceil(2 * ((z_alpha + z_beta) / d) ** 2)

    return PowerResult(
        sample_size_per_group=n,
        total_sample_size=n * 2,
        alpha=alpha,
        power=power,
        mde=mde,
        effect_size=d,
        test_type="Two-sample t-test (means)",
        alternative=alternative,
    )


def mde_from_sample_size(
    n_per_group: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> float:
    """
    Given a fixed sample size, compute the MDE you can reliably detect.

    Parameters
    ----------
    n_per_group : int
        Users per arm.
    baseline_rate : float
        Control conversion rate.

    Returns
    -------
    float : Minimum detectable effect (absolute change in rate).

    Examples
    --------
    >>> mde = mde_from_sample_size(n_per_group=5000, baseline_rate=0.10)
    >>> print(f"MDE: {mde:.4f}")
    """
    z_alpha = _z_alpha(alpha, alternative)
    z_beta = stats.norm.ppf(power)

    # Binary search for MDE
    lo, hi = 1e-6, 1 - baseline_rate
    for _ in range(100):
        mid = (lo + hi) / 2
        result = sample_size_proportion(baseline_rate, mid, alpha, power, alternative)
        if result.sample_size_per_group <= n_per_group:
            hi = mid
        else:
            lo = mid
    return round(mid, 6)


def power_curve(
    baseline_rate: float,
    mde_values: list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, list]:
    """
    Compute power vs sample size for multiple MDE values.
    Useful for plotting tradeoff curves.

    Returns a dict with keys: 'n', 'mde', 'power'.
    """
    results = {"n": [], "mde": [], "power": []}
    for mde in mde_values:
        for pwr in np.arange(0.5, 0.99, 0.05):
            res = sample_size_proportion(baseline_rate, mde, alpha, pwr, alternative)
            results["n"].append(res.sample_size_per_group)
            results["mde"].append(mde)
            results["power"].append(pwr)
    return results


# ── Internals ──────────────────────────────────────────────────────────────────

def _z_alpha(alpha: float, alternative: str) -> float:
    if alternative == "two-sided":
        return stats.norm.ppf(1 - alpha / 2)
    return stats.norm.ppf(1 - alpha)
