"""
frequentist.py — Classical (frequentist) hypothesis tests for A/B experiments.

Covers:
  - Two-proportion z-test (conversion rates)
  - Welch's t-test (continuous metrics)
  - Mann-Whitney U test (non-parametric fallback)
  - CUPED variance reduction (pre-experiment covariate adjustment)
  - Multiple testing correction (Bonferroni, Benjamini-Hochberg)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    alternative: str

    # Observed values
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift_pct: float

    # Confidence interval for the difference
    ci_lower: float
    ci_upper: float
    ci_level: float

    # Decision
    is_significant: bool
    verdict: str

    # Optional extras
    control_n: int = 0
    treatment_n: int = 0
    variance_reduction_pct: Optional[float] = None

    def __str__(self) -> str:
        lines = [
            "=" * 55,
            f"  {self.test_name}",
            "=" * 55,
            f"  Control   : mean={self.control_mean:.4f}  n={self.control_n:,}",
            f"  Treatment : mean={self.treatment_mean:.4f}  n={self.treatment_n:,}",
            f"  Absolute lift    : {self.absolute_lift:+.4f}",
            f"  Relative lift    : {self.relative_lift_pct:+.2f}%",
            f"  {int(self.ci_level*100)}% CI          : "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"  Test statistic   : {self.statistic:.4f}",
            f"  p-value          : {self.p_value:.4f}",
            f"  Alpha (α)        : {self.alpha}",
            f"  Significant?     : {'✅ YES' if self.is_significant else '❌ NO'}",
            f"  Verdict          : {self.verdict}",
        ]
        if self.variance_reduction_pct is not None:
            lines.append(
                f"  Variance reduced : {self.variance_reduction_pct:.1f}%"
            )
        lines.append("=" * 55)
        return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def proportion_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> TestResult:
    """
    Two-proportion z-test for conversion rate experiments.

    Parameters
    ----------
    control_conversions : int
        Number of successes in control.
    control_n : int
        Total users in control.
    treatment_conversions : int
        Number of successes in treatment.
    treatment_n : int
        Total users in treatment.
    alpha : float
        Significance level.
    alternative : str
        'two-sided', 'greater' (treatment > control), or 'less'.

    Returns
    -------
    TestResult

    Examples
    --------
    >>> result = proportion_test(500, 5000, 560, 5000)
    >>> print(result)
    """
    p_ctrl = control_conversions / control_n
    p_trt = treatment_conversions / treatment_n

    # Pooled proportion under H0
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))

    z = (p_trt - p_ctrl) / se
    p_value = _p_from_z(z, alternative)

    # CI for difference (unpooled SE)
    se_diff = np.sqrt(
        p_ctrl * (1 - p_ctrl) / control_n + p_trt * (1 - p_trt) / treatment_n
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_trt - p_ctrl
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    is_sig = p_value < alpha
    rel_lift = (diff / p_ctrl) * 100 if p_ctrl != 0 else float("inf")

    return TestResult(
        test_name="Two-Proportion Z-Test",
        statistic=round(z, 4),
        p_value=round(p_value, 4),
        alpha=alpha,
        alternative=alternative,
        control_mean=round(p_ctrl, 4),
        treatment_mean=round(p_trt, 4),
        absolute_lift=round(diff, 4),
        relative_lift_pct=round(rel_lift, 2),
        ci_lower=round(ci[0], 4),
        ci_upper=round(ci[1], 4),
        ci_level=1 - alpha,
        is_significant=is_sig,
        verdict=_verdict(is_sig, diff, alpha),
        control_n=control_n,
        treatment_n=treatment_n,
    )


def means_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    use_welch: bool = True,
) -> TestResult:
    """
    Welch's t-test (or Student's t-test) for continuous metrics.

    Parameters
    ----------
    control : array-like
        Raw observations for control group.
    treatment : array-like
        Raw observations for treatment group.
    alpha : float
        Significance level.
    alternative : str
        'two-sided', 'greater', or 'less'.
    use_welch : bool
        If True (default), use Welch's t-test (unequal variance). 
        More robust than Student's t-test.

    Examples
    --------
    >>> import numpy as np
    >>> ctrl = np.random.normal(50, 20, 5000)
    >>> trt  = np.random.normal(53, 20, 5000)
    >>> result = means_test(ctrl, trt)
    >>> print(result)
    """
    ctrl = np.asarray(control, dtype=float)
    trt = np.asarray(treatment, dtype=float)

    scipy_alt = {"two-sided": "two-sided", "greater": "greater", "less": "less"}[
        alternative
    ]

    t_stat, p_value = stats.ttest_ind(trt, ctrl, equal_var=not use_welch, alternative=scipy_alt)

    # CI for difference in means
    diff = trt.mean() - ctrl.mean()
    se_diff = np.sqrt(trt.var(ddof=1) / len(trt) + ctrl.var(ddof=1) / len(ctrl))
    df = _welch_df(ctrl, trt) if use_welch else len(ctrl) + len(trt) - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ci = (diff - t_crit * se_diff, diff + t_crit * se_diff)

    is_sig = p_value < alpha
    rel_lift = (diff / ctrl.mean()) * 100 if ctrl.mean() != 0 else float("inf")

    return TestResult(
        test_name=f"{'Welch' if use_welch else 'Student'}'s T-Test",
        statistic=round(t_stat, 4),
        p_value=round(p_value, 4),
        alpha=alpha,
        alternative=alternative,
        control_mean=round(float(ctrl.mean()), 4),
        treatment_mean=round(float(trt.mean()), 4),
        absolute_lift=round(diff, 4),
        relative_lift_pct=round(rel_lift, 2),
        ci_lower=round(ci[0], 4),
        ci_upper=round(ci[1], 4),
        ci_level=1 - alpha,
        is_significant=is_sig,
        verdict=_verdict(is_sig, diff, alpha),
        control_n=len(ctrl),
        treatment_n=len(trt),
    )


def cuped_test(
    control: np.ndarray,
    treatment: np.ndarray,
    control_pre: np.ndarray,
    treatment_pre: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

    Regresses out pre-experiment covariate to reduce metric variance,
    giving you more statistical power without collecting more data.
    Popularized by Microsoft Research (Deng et al., 2013).

    Parameters
    ----------
    control / treatment : array-like
        Post-experiment metric (e.g., revenue during the experiment).
    control_pre / treatment_pre : array-like
        Pre-experiment covariate (e.g., revenue in the 2 weeks before experiment).
        Must be the SAME metric, from before the experiment started.

    Returns
    -------
    TestResult with variance_reduction_pct populated.

    Examples
    --------
    >>> # Pre-experiment revenue correlates with in-experiment revenue
    >>> ctrl_pre = np.random.normal(45, 18, 5000)
    >>> trt_pre  = np.random.normal(45, 18, 5000)
    >>> ctrl     = ctrl_pre * 1.1 + np.random.normal(0, 5, 5000)
    >>> trt      = trt_pre  * 1.1 + np.random.normal(2, 5, 5000)  # +2 treatment effect
    >>> result   = cuped_test(ctrl, trt, ctrl_pre, trt_pre)
    >>> print(result)
    """
    ctrl = np.asarray(control, dtype=float)
    trt = np.asarray(treatment, dtype=float)
    ctrl_pre = np.asarray(control_pre, dtype=float)
    trt_pre = np.asarray(treatment_pre, dtype=float)

    all_post = np.concatenate([ctrl, trt])
    all_pre = np.concatenate([ctrl_pre, trt_pre])

    # Estimate theta: coefficient that minimises variance of adjusted metric
    theta = np.cov(all_post, all_pre)[0, 1] / np.var(all_pre, ddof=1)
    pre_grand_mean = all_pre.mean()

    # CUPED-adjusted outcomes
    ctrl_adj = ctrl - theta * (ctrl_pre - pre_grand_mean)
    trt_adj = trt - theta * (trt_pre - pre_grand_mean)

    # Variance reduction
    var_original = np.var(np.concatenate([ctrl, trt]), ddof=1)
    var_adjusted = np.var(np.concatenate([ctrl_adj, trt_adj]), ddof=1)
    var_reduction_pct = (1 - var_adjusted / var_original) * 100

    # Run standard t-test on adjusted values
    result = means_test(ctrl_adj, trt_adj, alpha=alpha, alternative=alternative)
    result.test_name = "CUPED Adjusted T-Test"
    result.variance_reduction_pct = round(var_reduction_pct, 1)

    # Keep original (unadjusted) means for interpretability
    result.control_mean = round(float(ctrl.mean()), 4)
    result.treatment_mean = round(float(trt.mean()), 4)
    result.absolute_lift = round(float(trt.mean() - ctrl.mean()), 4)
    result.relative_lift_pct = round(
        (result.absolute_lift / result.control_mean) * 100, 2
    )

    return result


def mannwhitney_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Mann-Whitney U test — non-parametric alternative to t-test.
    Use when metric is heavily skewed (e.g., revenue with large outliers).

    Examples
    --------
    >>> ctrl = np.random.exponential(50, 5000)   # skewed revenue
    >>> trt  = np.random.exponential(53, 5000)
    >>> result = mannwhitney_test(ctrl, trt)
    >>> print(result)
    """
    ctrl = np.asarray(control, dtype=float)
    trt = np.asarray(treatment, dtype=float)

    scipy_alt = {"two-sided": "two-sided", "greater": "greater", "less": "less"}[
        alternative
    ]
    u_stat, p_value = stats.mannwhitneyu(trt, ctrl, alternative=scipy_alt)

    diff = trt.mean() - ctrl.mean()
    is_sig = p_value < alpha

    return TestResult(
        test_name="Mann-Whitney U Test (Non-Parametric)",
        statistic=round(u_stat, 4),
        p_value=round(p_value, 4),
        alpha=alpha,
        alternative=alternative,
        control_mean=round(float(ctrl.mean()), 4),
        treatment_mean=round(float(trt.mean()), 4),
        absolute_lift=round(diff, 4),
        relative_lift_pct=round((diff / ctrl.mean()) * 100, 2) if ctrl.mean() != 0 else 0,
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        ci_level=1 - alpha,
        is_significant=is_sig,
        verdict=_verdict(is_sig, diff, alpha),
        control_n=len(ctrl),
        treatment_n=len(trt),
    )


def correct_multiple_tests(
    p_values: list[float],
    alpha: float = 0.05,
    method: Literal["bonferroni", "bh"] = "bh",
) -> pd.DataFrame:
    """
    Apply multiple testing correction to a list of p-values.

    Parameters
    ----------
    p_values : list of float
        Raw p-values from individual tests.
    alpha : float
        Family-wise error rate.
    method : str
        'bonferroni' — conservative, controls FWER.
        'bh' — Benjamini-Hochberg, controls FDR (recommended for DS).

    Returns
    -------
    pd.DataFrame with columns: test_index, raw_p, adjusted_p, significant

    Examples
    --------
    >>> ps = [0.01, 0.04, 0.20, 0.03, 0.15]
    >>> print(correct_multiple_tests(ps, method='bh'))
    """
    n = len(p_values)
    p_arr = np.array(p_values)

    if method == "bonferroni":
        adjusted = np.minimum(p_arr * n, 1.0)
    elif method == "bh":
        # Benjamini-Hochberg procedure
        order = np.argsort(p_arr)
        ranks = np.arange(1, n + 1)
        adjusted_ordered = np.minimum.accumulate(
            (p_arr[order] * n / ranks)[::-1]
        )[::-1]
        adjusted = np.empty(n)
        adjusted[order] = adjusted_ordered
        adjusted = np.minimum(adjusted, 1.0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bonferroni' or 'bh'.")

    return pd.DataFrame(
        {
            "test_index": range(n),
            "raw_p": p_arr,
            "adjusted_p": np.round(adjusted, 4),
            "significant": adjusted < alpha,
        }
    )


# ── Internals ──────────────────────────────────────────────────────────────────

def _p_from_z(z: float, alternative: str) -> float:
    if alternative == "two-sided":
        return 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        return 1 - stats.norm.cdf(z)
    else:
        return stats.norm.cdf(z)


def _welch_df(ctrl: np.ndarray, trt: np.ndarray) -> float:
    v1, v2 = ctrl.var(ddof=1), trt.var(ddof=1)
    n1, n2 = len(ctrl), len(trt)
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    return num / denom


def _verdict(is_sig: bool, lift: float, alpha: float) -> str:
    direction = "positive" if lift >= 0 else "negative"
    if is_sig:
        return f"Statistically significant {direction} effect detected."
    return "No significant effect detected. Fail to reject H0."
