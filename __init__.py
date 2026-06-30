"""
abtest — A/B Test Framework & Causal Inference Library
======================================================

Modules
-------
power       : Sample size & MDE calculations
frequentist : z-test, t-test, CUPED, multiple testing correction
bayesian    : Beta-Binomial and Normal-Normal Bayesian tests
sequential  : SPRT, always-valid p-values, novelty effect detection

Quick Start
-----------
>>> from abtest.power import sample_size_proportion
>>> from abtest.frequentist import proportion_test, cuped_test
>>> from abtest.bayesian import bayesian_proportion_test
>>> from abtest.sequential import SPRTMonitor, detect_novelty_effect, check_srm
"""

from .power import (
    sample_size_proportion,
    sample_size_means,
    mde_from_sample_size,
    power_curve,
    PowerResult,
)
from .frequentist import (
    proportion_test,
    means_test,
    cuped_test,
    mannwhitney_test,
    correct_multiple_tests,
    TestResult,
)
from .bayesian import (
    bayesian_proportion_test,
    bayesian_means_test,
    BayesianResult,
)
from .sequential import (
    SPRTMonitor,
    SPRTResult,
    always_valid_pvalue,
    detect_novelty_effect,
    NoveltyResult,
    check_srm,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = [
    # Power
    "sample_size_proportion",
    "sample_size_means",
    "mde_from_sample_size",
    "power_curve",
    "PowerResult",
    # Frequentist
    "proportion_test",
    "means_test",
    "cuped_test",
    "mannwhitney_test",
    "correct_multiple_tests",
    "TestResult",
    # Bayesian
    "bayesian_proportion_test",
    "bayesian_means_test",
    "BayesianResult",
    # Sequential
    "SPRTMonitor",
    "SPRTResult",
    "always_valid_pvalue",
    "detect_novelty_effect",
    "NoveltyResult",
    "check_srm",
]
