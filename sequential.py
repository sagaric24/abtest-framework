"""
sequential.py — Sequential testing & novelty effect detection.

The "peeking problem": repeatedly checking p-values inflates Type I error.
This module provides valid alternatives:

1. Sequential Probability Ratio Test (SPRT) — Wald's classic method.
2. Always-Valid Inference (mSPRT) — continuous monitoring with valid α.
3. Novelty Effect Detector — flags experiments where early treatment
   lift decays over time (users excited by newness, not real value).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class SPRTResult:
    """Result of a sequential test at the current sample size."""
    decision: Literal["accept_h0", "accept_h1", "continue"]
    llr: float                     # Log-likelihood ratio
    lower_boundary: float          # Stop if LLR < log(β / (1-α))
    upper_boundary: float          # Stop if LLR > log((1-β) / α)
    samples_seen: int
    current_rate_ctrl: float
    current_rate_trt: float
    message: str

    def __str__(self) -> str:
        icons = {"accept_h0": "❌", "accept_h1": "✅", "continue": "⏳"}
        return (
            f"SPRT Sequential Test @ n={self.samples_seen:,}/arm\n"
            f"  LLR = {self.llr:.3f}  "
            f"[{self.lower_boundary:.3f}, {self.upper_boundary:.3f}]\n"
            f"  Decision : {icons[self.decision]} {self.message}"
        )


@dataclass
class NoveltyResult:
    """Result of novelty/primacy effect analysis."""
    has_novelty_effect: bool
    early_lift: float          # Average lift in first window
    late_lift: float           # Average lift in last window
    lift_decay_pct: float      # How much lift decayed (%)
    p_value_decay: float       # p-value for decay trend (linear regression)
    window_lifts: list[float]  # Lift at each time window
    window_labels: list[str]
    recommendation: str

    def __str__(self) -> str:
        flag = "⚠️  NOVELTY EFFECT DETECTED" if self.has_novelty_effect else "✅ No novelty effect"
        return (
            f"{flag}\n"
            f"  Early lift  : {self.early_lift:+.4f}\n"
            f"  Late lift   : {self.late_lift:+.4f}\n"
            f"  Lift decay  : {self.lift_decay_pct:.1f}%\n"
            f"  Trend p-val : {self.p_value_decay:.4f}\n"
            f"  Recommendation: {self.recommendation}"
        )


# ── SPRT — Sequential Probability Ratio Test ───────────────────────────────────

class SPRTMonitor:
    """
    Online SPRT for binomial data (conversion experiments).

    Designed for use with a data stream — call `.update()` each time
    a new observation arrives. Stops as soon as a decision boundary is crossed.

    Parameters
    ----------
    p0 : float
        Null hypothesis rate (baseline conversion).
    p1 : float
        Alternative hypothesis rate (p0 + MDE).
    alpha : float
        Type I error tolerance (false positive rate).
    beta : float
        Type II error tolerance (false negative rate = 1 - power).

    Examples
    --------
    >>> monitor = SPRTMonitor(p0=0.10, p1=0.12, alpha=0.05, beta=0.20)
    >>> for ctrl_obs, trt_obs in data_stream:
    ...     result = monitor.update(ctrl_obs, trt_obs)
    ...     if result.decision != 'continue':
    ...         print(result)
    ...         break
    """

    def __init__(
        self,
        p0: float,
        p1: float,
        alpha: float = 0.05,
        beta: float = 0.20,
    ):
        self.p0 = p0
        self.p1 = p1
        self.alpha = alpha
        self.beta = beta

        # Wald boundaries
        self.upper = np.log((1 - beta) / alpha)    # Reject H0 if LLR > upper
        self.lower = np.log(beta / (1 - alpha))    # Accept H0 if LLR < lower

        self._llr = 0.0
        self._n_ctrl = 0
        self._n_trt = 0
        self._conv_ctrl = 0
        self._conv_trt = 0
        self.history: list[dict] = []

    def update(
        self, ctrl_conversion: int, trt_conversion: int
    ) -> SPRTResult:
        """
        Process one observation pair (one from each arm).

        Parameters
        ----------
        ctrl_conversion : int  0 or 1 (did this user convert in control?)
        trt_conversion  : int  0 or 1 (did this user convert in treatment?)
        """
        self._n_ctrl += 1
        self._n_trt += 1
        self._conv_ctrl += ctrl_conversion
        self._conv_trt += trt_conversion

        # Log-likelihood ratio update (sequential Bayes factor approximation)
        # Under H1: treatment has rate p1, control has rate p0
        # Under H0: both have rate p0
        if trt_conversion == 1:
            self._llr += np.log(self.p1 / self.p0)
        else:
            self._llr += np.log((1 - self.p1) / (1 - self.p0))

        # Decision
        if self._llr >= self.upper:
            decision = "accept_h1"
            msg = f"Significant effect detected (LLR ≥ {self.upper:.2f}). Ship treatment."
        elif self._llr <= self.lower:
            decision = "accept_h0"
            msg = f"No effect (LLR ≤ {self.lower:.2f}). Keep control."
        else:
            decision = "continue"
            msg = "Continue collecting data."

        result = SPRTResult(
            decision=decision,
            llr=round(self._llr, 4),
            lower_boundary=round(self.lower, 4),
            upper_boundary=round(self.upper, 4),
            samples_seen=self._n_ctrl,
            current_rate_ctrl=round(self._conv_ctrl / max(self._n_ctrl, 1), 4),
            current_rate_trt=round(self._conv_trt / max(self._n_trt, 1), 4),
            message=msg,
        )

        self.history.append(
            {
                "n": self._n_ctrl,
                "llr": self._llr,
                "decision": decision,
                "ctrl_rate": result.current_rate_ctrl,
                "trt_rate": result.current_rate_trt,
            }
        )
        return result

    def run_batch(self, ctrl_conversions: np.ndarray, trt_conversions: np.ndarray) -> SPRTResult:
        """Run SPRT over arrays of 0/1 observations. Returns final result."""
        result = None
        for c, t in zip(ctrl_conversions, trt_conversions):
            result = self.update(int(c), int(t))
            if result.decision != "continue":
                break
        return result

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ── Always-Valid Inference (mSPRT) ─────────────────────────────────────────────

def always_valid_pvalue(
    ctrl_conversions: np.ndarray,
    trt_conversions: np.ndarray,
    theta: float = 1.0,
) -> pd.Series:
    """
    Compute always-valid p-values (mSPRT / mixture sequential ratio test).

    Unlike classical p-values, these are valid at every look — you can
    monitor them continuously without inflating Type I error.

    Reference: Johari et al. (2017), "Peeking at A/B Tests"

    Parameters
    ----------
    ctrl_conversions, trt_conversions : array of 0/1
        Sequential observations, in time order.
    theta : float
        Variance parameter for the mixture prior. Larger = less conservative.

    Returns
    -------
    pd.Series : always-valid p-value at each time step.

    Examples
    --------
    >>> ctrl = np.random.binomial(1, 0.10, 5000)
    >>> trt  = np.random.binomial(1, 0.12, 5000)
    >>> avp = always_valid_pvalue(ctrl, trt)
    >>> print(avp.tail())
    """
    ctrl = np.asarray(ctrl_conversions, dtype=float)
    trt = np.asarray(trt_conversions, dtype=float)
    n = len(ctrl)

    pvalues = []
    for t in range(1, n + 1):
        c_obs = ctrl[:t]
        t_obs = trt[:t]

        s_c = c_obs.sum()
        s_t = t_obs.sum()
        n_t = t

        # Estimate rates
        p_hat = (s_c + s_t) / (2 * n_t)
        if p_hat == 0 or p_hat == 1:
            pvalues.append(1.0)
            continue

        # Variance of each arm's mean under H0
        v = p_hat * (1 - p_hat) / n_t

        # Test statistic (difference in means standardised)
        diff = (s_t - s_c) / n_t
        se = np.sqrt(2 * v)
        if se == 0:
            pvalues.append(1.0)
            continue

        # mSPRT p-value: analytic formula for mixture with Normal approx
        # Implements: p = sqrt(v/(v + theta)) * exp(-diff^2/(2*(v+theta)))  / normal_pdf
        # Simplified always-valid form:
        ratio = v / (v + theta)
        lr = np.sqrt(ratio) * np.exp(diff**2 / (2 * v) * (1 - ratio))
        p = min(1.0, 1.0 / max(lr, 1e-10))
        pvalues.append(p)

    return pd.Series(pvalues, name="always_valid_pvalue")


# ── Novelty Effect Detection ───────────────────────────────────────────────────

def detect_novelty_effect(
    ctrl_time_series: pd.DataFrame,
    trt_time_series: pd.DataFrame,
    date_col: str = "date",
    metric_col: str = "conversion_rate",
    n_windows: int = 4,
    decay_threshold: float = 0.30,
    trend_alpha: float = 0.10,
) -> NoveltyResult:
    """
    Detect novelty (primacy) effect: treatment lift that is high early
    and decays over time as users habituate to the change.

    How it works
    ------------
    1. Splits the experiment timeline into equal-width windows.
    2. Computes per-window lift (treatment - control).
    3. Fits a linear regression of lift ~ time_window.
    4. Flags novelty if: slope is significantly negative AND
       lift decays by more than `decay_threshold` from first to last window.

    Parameters
    ----------
    ctrl_time_series, trt_time_series : pd.DataFrame
        One row per day/period. Must have `date_col` and `metric_col`.
    n_windows : int
        Number of time windows to split the experiment into. Default 4.
    decay_threshold : float
        Minimum relative decay to flag as novelty (e.g., 0.30 = 30% decay).
    trend_alpha : float
        Significance level for the slope test.

    Returns
    -------
    NoveltyResult

    Examples
    --------
    >>> dates = pd.date_range("2024-01-01", periods=28)
    >>> # Treatment effect decays from 5% to 1% over 4 weeks
    >>> ctrl_df = pd.DataFrame({"date": dates, "conversion_rate": [0.10]*28})
    >>> trt_rates = [0.15]*7 + [0.13]*7 + [0.11]*7 + [0.10]*7
    >>> trt_df  = pd.DataFrame({"date": dates, "conversion_rate": trt_rates})
    >>> result = detect_novelty_effect(ctrl_df, trt_df)
    >>> print(result)
    """
    ctrl_df = ctrl_time_series.copy().sort_values(date_col).reset_index(drop=True)
    trt_df = trt_time_series.copy().sort_values(date_col).reset_index(drop=True)

    n_periods = len(ctrl_df)
    window_size = n_periods // n_windows

    window_lifts = []
    window_labels = []

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else n_periods

        ctrl_window = ctrl_df.iloc[start:end][metric_col].mean()
        trt_window = trt_df.iloc[start:end][metric_col].mean()
        lift = trt_window - ctrl_window

        window_lifts.append(round(lift, 5))
        window_labels.append(f"Week {w+1}" if window_size == 7 else f"Window {w+1}")

    # Linear regression: lift ~ window index
    x = np.arange(n_windows, dtype=float)
    y = np.array(window_lifts)
    slope, intercept, r_value, p_value, se = stats.linregress(x, y)

    early_lift = window_lifts[0]
    late_lift = window_lifts[-1]

    # Relative decay: (early - late) / |early|
    if abs(early_lift) < 1e-9:
        lift_decay_pct = 0.0
    else:
        lift_decay_pct = (early_lift - late_lift) / abs(early_lift) * 100

    has_novelty = (
        slope < 0                     # Declining trend
        and p_value < trend_alpha     # Trend is significant
        and lift_decay_pct > (decay_threshold * 100)  # Substantial decay
    )

    if has_novelty:
        recommendation = (
            "Extend the experiment. Early lift is likely inflated by novelty. "
            "Wait until lift stabilises before making a ship decision."
        )
    elif early_lift <= 0 and late_lift <= 0:
        recommendation = "Treatment shows no positive lift across all windows."
    else:
        recommendation = (
            "Lift appears stable across windows. Safe to use full-experiment results."
        )

    return NoveltyResult(
        has_novelty_effect=has_novelty,
        early_lift=early_lift,
        late_lift=late_lift,
        lift_decay_pct=round(lift_decay_pct, 1),
        p_value_decay=round(p_value, 4),
        window_lifts=window_lifts,
        window_labels=window_labels,
        recommendation=recommendation,
    )


# ── Sample Ratio Mismatch ──────────────────────────────────────────────────────

def check_srm(
    actual_ctrl: int,
    actual_trt: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.01,
) -> dict:
    """
    Sample Ratio Mismatch (SRM) check — critical sanity check before
    analysing any experiment.

    If actual split differs significantly from intended split, the
    randomisation is broken and results are invalid.

    Parameters
    ----------
    actual_ctrl, actual_trt : int
        Actual number of users assigned to each arm.
    expected_ratio : float
        Intended fraction in treatment. Default 0.5 (50/50 split).
    alpha : float
        Significance level for chi-square test. Default 0.01 (strict).

    Returns
    -------
    dict with keys: has_srm, chi2, p_value, actual_ratio, expected_ratio

    Examples
    --------
    >>> srm = check_srm(actual_ctrl=5000, actual_trt=4800)
    >>> if srm['has_srm']:
    ...     print("⚠️ SRM detected! Do not analyse this experiment.")
    """
    total = actual_ctrl + actual_trt
    expected_ctrl = total * (1 - expected_ratio)
    expected_trt = total * expected_ratio

    chi2, p_value = stats.chisquare(
        [actual_ctrl, actual_trt],
        f_exp=[expected_ctrl, expected_trt],
    )

    actual_ratio = actual_trt / total

    return {
        "has_srm": bool(p_value < alpha),
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "actual_ratio": round(actual_ratio, 4),
        "expected_ratio": expected_ratio,
        "actual_ctrl": actual_ctrl,
        "actual_trt": actual_trt,
        "message": (
            f"⚠️  SRM DETECTED (p={p_value:.4f}). "
            f"Actual split {actual_ratio:.1%} ≠ expected {expected_ratio:.1%}. "
            "Do not analyse this experiment."
            if p_value < alpha
            else f"✅ No SRM (p={p_value:.4f}). Split looks healthy."
        ),
    }
