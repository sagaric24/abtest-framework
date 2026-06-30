"""
examples/full_experiment_walkthrough.py
========================================
Complete walkthrough of a real-world A/B experiment using every
module in the abtest framework.

Scenario
--------
  Product: A SaaS checkout flow
  Hypothesis: A simplified checkout (treatment) increases conversion vs 
              current flow (control).
  Primary metric: Conversion rate
  Secondary metric: Revenue per user
  Guardrail metric: Session duration (should not decrease)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from abtest.power import sample_size_proportion, mde_from_sample_size
from abtest.frequentist import (
    proportion_test, means_test, cuped_test, correct_multiple_tests
)
from abtest.bayesian import bayesian_proportion_test, bayesian_means_test
from abtest.sequential import (
    SPRTMonitor, always_valid_pvalue, detect_novelty_effect, check_srm
)

rng = np.random.default_rng(42)

print("\n" + "━" * 60)
print("  A/B TEST FRAMEWORK — Full Experiment Walkthrough")
print("━" * 60)


# ────────────────────────────────────────────────────────────
# STEP 1: PRE-EXPERIMENT — Power Analysis
# ────────────────────────────────────────────────────────────
print("\n📐 STEP 1: Power Analysis")
print("-" * 40)

baseline_cr = 0.10           # 10% current conversion rate
mde = 0.02                   # Want to detect +2pp lift minimum

power_result = sample_size_proportion(
    baseline_rate=baseline_cr,
    mde=mde,
    alpha=0.05,
    power=0.80,
    alternative="two-sided",
)
print(power_result)

# What if we can only run for 2 weeks and expect 4,000 users/arm?
available_n = 4000
detectable_mde = mde_from_sample_size(
    n_per_group=available_n,
    baseline_rate=baseline_cr,
)
print(f"\nWith {available_n:,} users/arm, smallest detectable MDE: "
      f"{detectable_mde:.4f} ({detectable_mde/baseline_cr:.1%} relative)")


# ────────────────────────────────────────────────────────────
# STEP 2: SIMULATE EXPERIMENT DATA
# ────────────────────────────────────────────────────────────
print("\n\n🎲 STEP 2: Simulating Experiment Data")
print("-" * 40)

N = power_result.sample_size_per_group
TRUE_EFFECT = 0.025   # True treatment effect

# Pre-experiment revenue (covariate for CUPED)
ctrl_pre_revenue = rng.normal(45, 18, N)
trt_pre_revenue = rng.normal(45, 18, N)

# Conversion: correlated with pre-revenue baseline
ctrl_p = baseline_cr
trt_p = baseline_cr + TRUE_EFFECT

ctrl_conversions_arr = rng.binomial(1, ctrl_p, N)
trt_conversions_arr = rng.binomial(1, trt_p, N)

# Revenue per user (only for converters, correlated with pre-revenue)
ctrl_revenue = (
    ctrl_conversions_arr * (ctrl_pre_revenue * 1.1 + rng.normal(0, 8, N))
)
trt_revenue = (
    trt_conversions_arr * (trt_pre_revenue * 1.1 + rng.normal(2, 8, N))
)

# Session duration (guardrail — no real effect)
ctrl_duration = rng.normal(180, 60, N)  # seconds
trt_duration = rng.normal(175, 60, N)   # slightly lower (new checkout is faster)

ctrl_conv_total = int(ctrl_conversions_arr.sum())
trt_conv_total = int(trt_conversions_arr.sum())

print(f"Simulated {N:,} users per arm")
print(f"Control conversions  : {ctrl_conv_total:,} / {N:,} = {ctrl_conv_total/N:.2%}")
print(f"Treatment conversions: {trt_conv_total:,} / {N:,} = {trt_conv_total/N:.2%}")


# ────────────────────────────────────────────────────────────
# STEP 3: SANITY CHECKS BEFORE ANALYSIS
# ────────────────────────────────────────────────────────────
print("\n\n🔍 STEP 3: Pre-Analysis Sanity Checks")
print("-" * 40)

# Sample Ratio Mismatch check
srm = check_srm(actual_ctrl=N, actual_trt=N)
print(f"SRM Check: {srm['message']}")

# A/A test simulation (should find ~5% false positives at α=0.05)
print("\nRunning 1,000 A/A tests to validate Type I error rate...")
false_positives = 0
for _ in range(1000):
    aa_ctrl = rng.binomial(1, baseline_cr, 1000)
    aa_trt = rng.binomial(1, baseline_cr, 1000)
    from scipy import stats
    _, p = stats.ttest_ind(aa_trt, aa_ctrl)
    if p < 0.05:
        false_positives += 1
print(f"False positive rate: {false_positives/1000:.1%} (expected ~5%)")


# ────────────────────────────────────────────────────────────
# STEP 4: FREQUENTIST ANALYSIS
# ────────────────────────────────────────────────────────────
print("\n\n📊 STEP 4: Frequentist Analysis")
print("-" * 40)

# Primary metric: conversion rate
print("\n[Primary] Conversion Rate:")
conv_result = proportion_test(
    control_conversions=ctrl_conv_total,
    control_n=N,
    treatment_conversions=trt_conv_total,
    treatment_n=N,
    alpha=0.05,
)
print(conv_result)

# Secondary metric: revenue (CUPED-adjusted)
print("\n[Secondary] Revenue per User — CUPED adjusted:")
cuped_result = cuped_test(
    control=ctrl_revenue,
    treatment=trt_revenue,
    control_pre=ctrl_pre_revenue,
    treatment_pre=trt_pre_revenue,
    alpha=0.05,
)
print(cuped_result)

# Guardrail: session duration
print("\n[Guardrail] Session Duration:")
duration_result = means_test(ctrl_duration, trt_duration, alpha=0.05)
print(duration_result)

# Multiple testing correction across all 3 metrics
print("\n[Multiple Testing] Benjamini-Hochberg correction:")
p_values = [conv_result.p_value, cuped_result.p_value, duration_result.p_value]
correction_df = correct_multiple_tests(p_values, method="bh")
correction_df.insert(0, "metric", ["conversion_rate", "revenue", "session_duration"])
print(correction_df.to_string(index=False))


# ────────────────────────────────────────────────────────────
# STEP 5: BAYESIAN ANALYSIS
# ────────────────────────────────────────────────────────────
print("\n\n🎯 STEP 5: Bayesian Analysis")
print("-" * 40)

print("\n[Primary] Bayesian Conversion Rate Test:")
bayes_result = bayesian_proportion_test(
    control_conversions=ctrl_conv_total,
    control_n=N,
    treatment_conversions=trt_conv_total,
    treatment_n=N,
    loss_threshold=0.001,
)
print(bayes_result)

print("\n[Secondary] Bayesian Revenue Test:")
bayes_rev = bayesian_means_test(ctrl_revenue, trt_revenue)
print(bayes_rev)


# ────────────────────────────────────────────────────────────
# STEP 6: SEQUENTIAL MONITORING SIMULATION
# ────────────────────────────────────────────────────────────
print("\n\n⏱  STEP 6: Sequential Testing (SPRT)")
print("-" * 40)
print("Simulating live monitoring — did SPRT stop early?\n")

monitor = SPRTMonitor(p0=baseline_cr, p1=baseline_cr + mde, alpha=0.05, beta=0.20)
final_result = monitor.run_batch(ctrl_conversions_arr, trt_conversions_arr)

history = monitor.history_df()
stopped_at = history[history["decision"] != "continue"]

if not stopped_at.empty:
    stop_n = int(stopped_at.iloc[0]["n"])
    savings = (N - stop_n) / N * 100
    print(f"SPRT stopped at n={stop_n:,} per arm "
          f"({savings:.0f}% fewer samples than fixed-horizon test)")
else:
    print(f"SPRT did not reach a boundary. Final result:")

print(final_result)


# ────────────────────────────────────────────────────────────
# STEP 7: NOVELTY EFFECT DETECTION
# ────────────────────────────────────────────────────────────
print("\n\n🔔 STEP 7: Novelty Effect Detection")
print("-" * 40)

# Simulate 28-day experiment with decaying novelty effect
dates = pd.date_range("2024-01-01", periods=28)
ctrl_daily_cr = 0.10 + rng.normal(0, 0.003, 28)
# Novelty: +5% in week 1, decaying to +2% by week 4
novelty_effect = np.array([0.05]*7 + [0.04]*7 + [0.025]*7 + [0.02]*7)
trt_daily_cr = ctrl_daily_cr + novelty_effect + rng.normal(0, 0.003, 28)

ctrl_ts = pd.DataFrame({"date": dates, "conversion_rate": ctrl_daily_cr})
trt_ts = pd.DataFrame({"date": dates, "conversion_rate": trt_daily_cr})

novelty_result = detect_novelty_effect(ctrl_ts, trt_ts, n_windows=4)
print(novelty_result)

window_df = pd.DataFrame({
    "window": novelty_result.window_labels,
    "lift": [f"{l:+.4f}" for l in novelty_result.window_lifts],
})
print("\nLift by window:")
print(window_df.to_string(index=False))


# ────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────
print("\n\n" + "━" * 60)
print("  EXPERIMENT SUMMARY")
print("━" * 60)

summary = {
    "Planned sample (per arm)": f"{N:,}",
    "Conversion — significant?": str(conv_result.is_significant),
    "Conversion — lift": f"{conv_result.relative_lift_pct:+.1f}%",
    "Revenue — significant?": str(cuped_result.is_significant),
    "Revenue — variance reduced": f"{cuped_result.variance_reduction_pct:.0f}%",
    "Session duration — guardrail ok?": str(not duration_result.is_significant),
    "P(treatment better) — Bayesian": f"{bayes_result.prob_treatment_better:.1%}",
    "Bayesian decision": bayes_result.decision.split("(")[0].strip(),
    "Novelty effect?": str(novelty_result.has_novelty_effect),
}

for k, v in summary.items():
    print(f"  {k:<40} {v}")
print("━" * 60)
print("\n✅ Walkthrough complete.\n")
