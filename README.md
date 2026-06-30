# abtest-framework

A production-grade A/B testing library for data scientists. Covers the full experiment lifecycle — from power analysis through Bayesian decision-making — with a Streamlit dashboard for interactive analysis.

## Features

| Module | What it does |
|--------|-------------|
| `power` | Sample size & MDE calculations for proportions and means |
| `frequentist` | z-test, Welch's t-test, CUPED variance reduction, Mann-Whitney, Bonferroni/BH correction |
| `bayesian` | Beta-Binomial & Normal-Normal Bayesian tests with expected loss |
| `sequential` | SPRT, always-valid p-values, novelty/primacy effect detection, SRM check |

## Quick Start

```python
from abtest.power import sample_size_proportion
from abtest.frequentist import proportion_test, cuped_test
from abtest.bayesian import bayesian_proportion_test
from abtest.sequential import SPRTMonitor, detect_novelty_effect, check_srm

# 1. Pre-experiment: how many users do I need?
result = sample_size_proportion(baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.80)
print(result)
# Per-group n: 3,841  |  Total n: 7,682

# 2. Sanity check before analysis
srm = check_srm(actual_ctrl=3841, actual_trt=3841)
print(srm["message"])  # ✅ No SRM detected

# 3. Frequentist test
result = proportion_test(388, 3841, 469, 3841, alpha=0.05)
print(result)
# Lift: +20.9%  |  p=0.0033  |  ✅ Significant

# 4. CUPED variance reduction (secondary metric)
cuped_result = cuped_test(ctrl_revenue, trt_revenue, ctrl_pre_revenue, trt_pre_revenue)
print(f"Variance reduced by {cuped_result.variance_reduction_pct}%")

# 5. Bayesian decision
bayes = bayesian_proportion_test(388, 3841, 469, 3841, loss_threshold=0.001)
print(f"P(treatment better): {bayes.prob_treatment_better:.1%}")
print(f"Decision: {bayes.decision}")
# P(treatment better): 99.8%
# Decision: ✅ Ship Treatment

# 6. Sequential monitoring (stop early when evidence is strong)
monitor = SPRTMonitor(p0=0.10, p1=0.12, alpha=0.05, beta=0.20)
for ctrl_obs, trt_obs in data_stream:
    result = monitor.update(ctrl_obs, trt_obs)
    if result.decision != "continue":
        print(result)
        break

# 7. Check for novelty effects
novelty = detect_novelty_effect(ctrl_timeseries, trt_timeseries, n_windows=4)
print(novelty)
```

## Installation

```bash
# From PyPI (once published)
pip install abtest-framework

# From source
git clone https://github.com/yourusername/abtest-framework
cd abtest-framework
pip install -e ".[app]"
```

## Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard has four pages:
- **Power Analysis** — interactive sample size calculator with power curves
- **Significance Tests** — proportion z-test, CUPED, multiple testing correction
- **Bayesian Analysis** — posterior visualisation and expected loss decisions
- **Sequential & Novelty** — SPRT monitoring chart and novelty effect detector

## Project Structure

```
abtest-framework/
├── abtest/
│   ├── __init__.py       # Clean public API
│   ├── power.py          # Sample size & MDE calculations
│   ├── frequentist.py    # z-test, t-test, CUPED, BH correction
│   ├── bayesian.py       # Beta-Binomial & Normal-Normal Bayesian tests
│   └── sequential.py     # SPRT, always-valid p-values, novelty detection
├── tests/
│   └── test_abtest.py    # 23 unit tests (pytest)
├── examples/
│   └── full_experiment_walkthrough.py
├── app.py                # Streamlit dashboard
├── pyproject.toml        # Package config (PyPI-ready)
└── README.md
```

## Key Concepts Implemented

### CUPED (Controlled-experiment Using Pre-Experiment Data)
Variance reduction technique from Microsoft Research (Deng et al., 2013). Regresses out a pre-experiment covariate to reduce metric noise, giving equivalent power with fewer users.

### Expected Loss (Bayesian Decision Rule)
Instead of a binary significant/not-significant decision, compute:
- `E[loss | choose treatment]` = average amount lost if treatment turns out worse
- Ship when expected loss falls below a business-defined threshold

### SPRT (Sequential Probability Ratio Test)
Wald's sequential test that lets you stop experiments early when evidence accumulates — without inflating the false positive rate the way repeated classical testing does.

### Novelty Effect Detection
Fits a linear regression on per-window treatment lift over time. Flags experiments where lift significantly decays — indicating users were reacting to novelty, not real value.

### Sample Ratio Mismatch (SRM)
Chi-square test that checks whether the actual user split matches the intended randomisation ratio. A significant SRM means the experiment has a bug and results are invalid.

## Testing

```bash
pytest tests/ -v
# 23 passed in 1.55s
```

## Publishing to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

## References

- Deng, A., et al. (2013). *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data.* WSDM.
- Johari, R., et al. (2017). *Peeking at A/B Tests: Why it matters, and what to do about it.* KDD.
- Wald, A. (1945). *Sequential Tests of Statistical Hypotheses.* Annals of Mathematical Statistics.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments.* Cambridge.

## License

MIT
