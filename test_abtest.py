"""
tests/test_abtest.py — Unit tests for the abtest framework.
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from abtest.power import sample_size_proportion, sample_size_means, mde_from_sample_size
from abtest.frequentist import proportion_test, means_test, cuped_test, correct_multiple_tests
from abtest.bayesian import bayesian_proportion_test, bayesian_means_test
from abtest.sequential import SPRTMonitor, detect_novelty_effect, check_srm


# ── Power ──────────────────────────────────────────────────────────────────────

class TestPower:
    def test_sample_size_proportion_basic(self):
        result = sample_size_proportion(0.10, 0.02, alpha=0.05, power=0.80)
        assert result.sample_size_per_group > 0
        assert result.total_sample_size == result.sample_size_per_group * 2
        # Known reference: ~3841 for these params
        assert 3500 < result.sample_size_per_group < 4500

    def test_higher_power_needs_larger_n(self):
        n80 = sample_size_proportion(0.10, 0.02, power=0.80).sample_size_per_group
        n90 = sample_size_proportion(0.10, 0.02, power=0.90).sample_size_per_group
        assert n90 > n80

    def test_smaller_mde_needs_larger_n(self):
        n_small = sample_size_proportion(0.10, 0.01).sample_size_per_group
        n_large = sample_size_proportion(0.10, 0.05).sample_size_per_group
        assert n_small > n_large

    def test_sample_size_means(self):
        result = sample_size_means(50.0, 20.0, 5.0)
        assert result.sample_size_per_group > 0
        assert result.effect_size == pytest.approx(0.25, abs=0.01)  # Cohen's d = 5/20

    def test_mde_from_sample_size_roundtrip(self):
        n = 5000
        baseline = 0.10
        mde = mde_from_sample_size(n, baseline)
        recovered_n = sample_size_proportion(baseline, mde).sample_size_per_group
        assert abs(recovered_n - n) / n < 0.05  # within 5%


# ── Frequentist ────────────────────────────────────────────────────────────────

class TestFrequentist:
    def test_proportion_significant(self):
        """Large effect should be significant."""
        result = proportion_test(500, 5000, 650, 5000)
        assert result.is_significant
        assert result.p_value < 0.05
        assert result.absolute_lift > 0

    def test_proportion_not_significant(self):
        """Tiny effect with small n should not be significant."""
        result = proportion_test(100, 1000, 101, 1000)
        assert not result.is_significant
        assert result.p_value > 0.05

    def test_proportion_ci_contains_zero_when_not_sig(self):
        result = proportion_test(100, 1000, 102, 1000)
        if not result.is_significant:
            assert result.ci_lower < 0 < result.ci_upper or result.ci_lower <= 0

    def test_means_test_detects_effect(self):
        rng = np.random.default_rng(1)
        ctrl = rng.normal(50, 10, 2000)
        trt = rng.normal(55, 10, 2000)
        result = means_test(ctrl, trt)
        assert result.is_significant
        assert result.absolute_lift == pytest.approx(trt.mean() - ctrl.mean(), abs=0.01)

    def test_cuped_reduces_variance(self):
        rng = np.random.default_rng(7)
        ctrl_pre = rng.normal(45, 18, 3000)
        trt_pre = rng.normal(45, 18, 3000)
        ctrl = ctrl_pre * 1.1 + rng.normal(0, 5, 3000)
        trt = trt_pre * 1.1 + rng.normal(2, 5, 3000)
        result = cuped_test(ctrl, trt, ctrl_pre, trt_pre)
        assert result.variance_reduction_pct is not None
        # With high pre/post correlation, variance should reduce
        assert result.variance_reduction_pct > 0

    def test_multiple_correction_bh_returns_dataframe(self):
        ps = [0.01, 0.04, 0.20, 0.03, 0.15]
        df = correct_multiple_tests(ps, method="bh")
        assert len(df) == len(ps)
        assert "adjusted_p" in df.columns
        assert "significant" in df.columns

    def test_multiple_correction_bonferroni(self):
        ps = [0.01, 0.04]
        df = correct_multiple_tests(ps, alpha=0.05, method="bonferroni")
        # Bonferroni: adjusted = p * n
        assert df.loc[0, "adjusted_p"] == pytest.approx(0.01 * 2, abs=0.001)

    def test_bh_less_conservative_than_bonferroni(self):
        """BH should declare more discoveries than Bonferroni."""
        ps = [0.01, 0.02, 0.03, 0.04, 0.05]
        bh = correct_multiple_tests(ps, method="bh")
        bon = correct_multiple_tests(ps, method="bonferroni")
        assert bh["significant"].sum() >= bon["significant"].sum()


# ── Bayesian ───────────────────────────────────────────────────────────────────

class TestBayesian:
    def test_bayesian_proportion_high_confidence(self):
        """Strong treatment effect → high P(B>A)."""
        result = bayesian_proportion_test(500, 5000, 700, 5000)
        assert result.prob_treatment_better > 0.99

    def test_bayesian_proportion_no_effect(self):
        """Same conversion → ~50% chance treatment is better."""
        result = bayesian_proportion_test(500, 5000, 500, 5000)
        assert 0.3 < result.prob_treatment_better < 0.7

    def test_bayesian_expected_loss_sums_to_positive(self):
        result = bayesian_proportion_test(500, 5000, 550, 5000)
        assert result.expected_loss_treatment >= 0
        assert result.expected_loss_control >= 0

    def test_bayesian_means_test_runs(self):
        rng = np.random.default_rng(3)
        ctrl = rng.normal(50, 20, 3000)
        trt = rng.normal(55, 20, 3000)
        result = bayesian_means_test(ctrl, trt)
        assert result.prob_treatment_better > 0.8


# ── Sequential ─────────────────────────────────────────────────────────────────

class TestSequential:
    def test_sprt_detects_real_effect(self):
        rng = np.random.default_rng(42)
        ctrl = rng.binomial(1, 0.10, 5000)
        trt = rng.binomial(1, 0.14, 5000)  # Large effect
        monitor = SPRTMonitor(p0=0.10, p1=0.12, alpha=0.05, beta=0.20)
        result = monitor.run_batch(ctrl, trt)
        assert result.decision == "accept_h1"

    def test_sprt_history_grows(self):
        rng = np.random.default_rng(10)
        ctrl = rng.binomial(1, 0.10, 100)
        trt = rng.binomial(1, 0.10, 100)  # No effect — should continue
        monitor = SPRTMonitor(p0=0.10, p1=0.15, alpha=0.05, beta=0.20)
        monitor.run_batch(ctrl, trt)
        assert len(monitor.history) <= 100

    def test_check_srm_no_mismatch(self):
        result = check_srm(5000, 5000)
        assert not result["has_srm"]

    def test_check_srm_detects_mismatch(self):
        # Very skewed split
        result = check_srm(9000, 1000, expected_ratio=0.5)
        assert result["has_srm"]

    def test_novelty_effect_detected(self):
        dates = pd.date_range("2024-01-01", periods=28)
        ctrl = pd.DataFrame({"date": dates, "conversion_rate": [0.10] * 28})
        # Sharp decay from 15% to 10%
        rates = [0.15] * 7 + [0.13] * 7 + [0.11] * 7 + [0.10] * 7
        trt = pd.DataFrame({"date": dates, "conversion_rate": rates})
        result = detect_novelty_effect(ctrl, trt, n_windows=4, decay_threshold=0.20)
        assert result.has_novelty_effect
        assert result.early_lift > result.late_lift

    def test_novelty_effect_not_detected_stable(self):
        dates = pd.date_range("2024-01-01", periods=28)
        ctrl = pd.DataFrame({"date": dates, "conversion_rate": [0.10] * 28})
        trt = pd.DataFrame({"date": dates, "conversion_rate": [0.12] * 28})  # Stable lift
        result = detect_novelty_effect(ctrl, trt, n_windows=4)
        assert not result.has_novelty_effect
