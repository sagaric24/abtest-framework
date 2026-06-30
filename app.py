"""
app.py — Streamlit dashboard for the abtest framework.
Run with: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from abtest.power import sample_size_proportion, mde_from_sample_size
from abtest.frequentist import proportion_test, means_test, cuped_test, correct_multiple_tests
from abtest.bayesian import bayesian_proportion_test
from abtest.sequential import SPRTMonitor, detect_novelty_effect, check_srm

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Test Framework",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4CAF50;
        margin-bottom: 0.5rem;
    }
    .warning-card {
        background: #fff8e1;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #ff9800;
        margin-bottom: 0.5rem;
    }
    .danger-card {
        background: #fce4ec;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #e53935;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🧪 A/B Test Framework")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📐 Power Analysis", "📊 Significance Tests", "🎯 Bayesian Analysis", "⏱ Sequential & Novelty"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · SciPy · statsmodels")
st.sidebar.caption("Portfolio project — Data Scientist / MLE")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: POWER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "📐 Power Analysis":
    st.title("📐 Power Analysis & Sample Size Calculator")
    st.markdown(
        "Determine **how many users you need** before running an experiment, "
        "or what **minimum effect you can detect** given a traffic constraint."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Parameters")
        baseline_rate = st.slider("Baseline conversion rate", 0.01, 0.50, 0.10, 0.005,
                                   format="%.3f")
        mde = st.slider("Min. detectable effect (absolute)", 0.001, 0.10, 0.02, 0.001,
                         format="%.3f")
        alpha = st.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)
        power = st.select_slider("Statistical power (1-β)", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                                  value=0.80)
        alternative = st.radio("Alternative hypothesis", ["two-sided", "greater", "less"],
                                horizontal=True)

    with col2:
        result = sample_size_proportion(baseline_rate, mde, alpha, power, alternative)
        st.subheader("Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Per-group n", f"{result.sample_size_per_group:,}")
        m2.metric("Total n", f"{result.total_sample_size:,}")
        m3.metric("Effect size (h)", f"{result.effect_size:.3f}")

        treatment_rate = baseline_rate + mde
        m4, m5, m6 = st.columns(3)
        m4.metric("Control rate", f"{baseline_rate:.1%}")
        m5.metric("Treatment rate", f"{treatment_rate:.1%}")
        m6.metric("Relative MDE", f"{mde/baseline_rate:.1%}")

    # Power curve
    st.markdown("---")
    st.subheader("Power vs Sample Size Curve")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: power vs n for different MDEs
    mde_vals = [mde * 0.5, mde, mde * 1.5, mde * 2.0]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    ax = axes[0]
    n_range = np.linspace(100, result.sample_size_per_group * 2.5, 200).astype(int)

    for m, color in zip(mde_vals, colors):
        powers = []
        for n in n_range:
            p_pool = (baseline_rate + baseline_rate + m) / 2
            se = np.sqrt(2 * p_pool * (1 - p_pool) / n)
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            pwr = 1 - stats.norm.cdf(z_alpha - abs(m) / se) + stats.norm.cdf(-z_alpha - abs(m) / se)
            powers.append(pwr)
        ax.plot(n_range, powers, color=color, lw=2, label=f"MDE={m:.3f} ({m/baseline_rate:.0%})")

    ax.axhline(power, ls="--", color="gray", alpha=0.7, label=f"Target power={power}")
    ax.axvline(result.sample_size_per_group, ls=":", color="gray", alpha=0.7)
    ax.set_xlabel("Sample size per group")
    ax.set_ylabel("Statistical Power")
    ax.set_title("Power vs Sample Size")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Right: MDE vs available N
    ax2 = axes[1]
    n_vals = np.linspace(500, result.sample_size_per_group * 3, 100).astype(int)
    mde_detectable = [mde_from_sample_size(int(n), baseline_rate, alpha, power) for n in n_vals]
    ax2.plot(n_vals, [m / baseline_rate * 100 for m in mde_detectable],
             color="#2980b9", lw=2.5)
    ax2.axvline(result.sample_size_per_group, ls="--", color="#e74c3c", alpha=0.8,
                label=f"Your n={result.sample_size_per_group:,}")
    ax2.set_xlabel("Sample size per group")
    ax2.set_ylabel("Detectable MDE (% relative)")
    ax2.set_title("MDE You Can Detect vs Available Traffic")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SIGNIFICANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Significance Tests":
    st.title("📊 Frequentist Significance Tests")

    tab1, tab2, tab3 = st.tabs(["Conversion Rate", "Continuous Metric + CUPED", "Multiple Tests"])

    # ── Tab 1: Proportion test ────────────────────────────────────────────────
    with tab1:
        st.subheader("Two-Proportion Z-Test")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Control**")
            ctrl_n = st.number_input("Users (control)", 100, 1000000, 5000, key="ctrl_n")
            ctrl_conv = st.number_input("Conversions (control)", 0, int(ctrl_n), 500, key="ctrl_c")

        with col2:
            st.markdown("**Treatment**")
            trt_n = st.number_input("Users (treatment)", 100, 1000000, 5000, key="trt_n")
            trt_conv = st.number_input("Conversions (treatment)", 0, int(trt_n), 560, key="trt_c")

        col3, col4 = st.columns(2)
        with col3:
            alpha_prop = st.select_slider("α", [0.01, 0.05, 0.10], value=0.05, key="alpha_prop")
        with col4:
            alt_prop = st.radio("Alternative", ["two-sided", "greater", "less"],
                                horizontal=True, key="alt_prop")

        # SRM check
        srm = check_srm(int(ctrl_n), int(trt_n))
        if srm["has_srm"]:
            st.markdown(f'<div class="danger-card">⚠️ <b>Sample Ratio Mismatch Detected!</b><br>{srm["message"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card">✅ No SRM detected — randomisation looks healthy.</div>',
                        unsafe_allow_html=True)

        result = proportion_test(int(ctrl_conv), int(ctrl_n), int(trt_conv), int(trt_n),
                                  alpha=alpha_prop, alternative=alt_prop)

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Control rate", f"{result.control_mean:.2%}")
        col6.metric("Treatment rate", f"{result.treatment_mean:.2%}")
        col7.metric("Relative lift", f"{result.relative_lift_pct:+.1f}%",
                    delta=f"{result.relative_lift_pct:+.1f}%")
        col8.metric("p-value", f"{result.p_value:.4f}",
                    delta="significant ✅" if result.is_significant else "not significant ❌",
                    delta_color="normal" if result.is_significant else "inverse")

        # Visualise CI
        fig, ax = plt.subplots(figsize=(9, 2.5))
        diff = result.absolute_lift
        ci_lo, ci_hi = result.ci_lower, result.ci_upper
        color = "#27ae60" if result.is_significant else "#7f8c8d"
        ax.barh(["Lift"], [diff], xerr=[[diff - ci_lo], [ci_hi - diff]],
                color=color, alpha=0.8, capsize=6, height=0.4)
        ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.5)
        ax.set_xlabel("Absolute lift (treatment − control)")
        ax.set_title(f"{int((1-alpha_prop)*100)}% Confidence Interval for Lift")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close()

        with st.expander("📖 Interpretation guide"):
            st.markdown(f"""
- **p = {result.p_value:.4f}** — probability of observing this lift (or larger) if there's truly no effect.
- **{"Reject" if result.is_significant else "Fail to reject"} H₀** at α={alpha_prop}.
- **{int((1-alpha_prop)*100)}% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]** — if CI excludes 0, result is significant.
- **Relative lift: {result.relative_lift_pct:+.2f}%** — how much better treatment is relative to control.
            """)

    # ── Tab 2: CUPED ──────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Continuous Metric Test with CUPED Variance Reduction")
        st.info("CUPED uses pre-experiment data to reduce metric variance, giving more "
                "statistical power without collecting more data.")

        col1, col2 = st.columns(2)
        with col1:
            ctrl_mean_input = st.number_input("Control mean (post)", value=50.0)
            ctrl_std_input = st.number_input("Control std (post)", value=20.0, min_value=0.1)
            ctrl_n2 = st.number_input("Control n", value=3000, min_value=100, key="cn2")
            pre_corr = st.slider("Pre/post correlation (ρ)", 0.0, 0.99, 0.60,
                                  help="How correlated is the pre-experiment covariate?")

        with col2:
            trt_lift = st.number_input("Treatment lift (absolute)", value=2.0)
            alpha_cont = st.select_slider("α", [0.01, 0.05, 0.10], value=0.05, key="alpha_cont")
            trt_n2 = st.number_input("Treatment n", value=3000, min_value=100, key="tn2")

        if st.button("Run Simulation & Test"):
            rng2 = np.random.default_rng(99)
            n = int(min(ctrl_n2, trt_n2))

            ctrl_pre = rng2.normal(ctrl_mean_input * 0.9, ctrl_std_input, n)
            trt_pre = rng2.normal(ctrl_mean_input * 0.9, ctrl_std_input, n)

            # Generate correlated post-experiment data
            ctrl_post = (pre_corr * ctrl_pre
                         + np.sqrt(1 - pre_corr**2) * rng2.normal(0, ctrl_std_input, n)
                         + ctrl_mean_input * (1 - pre_corr))
            trt_post = (pre_corr * trt_pre
                        + np.sqrt(1 - pre_corr**2) * rng2.normal(0, ctrl_std_input, n)
                        + (ctrl_mean_input + trt_lift) * (1 - pre_corr) + trt_lift * pre_corr)

            plain = means_test(ctrl_post, trt_post, alpha=alpha_cont)
            cuped = cuped_test(ctrl_post, trt_post, ctrl_pre, trt_pre, alpha=alpha_cont)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Standard T-Test**")
                st.metric("p-value", f"{plain.p_value:.4f}")
                st.metric("Significant?", "✅ YES" if plain.is_significant else "❌ NO")
            with col4:
                st.markdown("**CUPED-Adjusted Test**")
                st.metric("p-value", f"{cuped.p_value:.4f}")
                st.metric("Significant?", "✅ YES" if cuped.is_significant else "❌ NO")
                if cuped.variance_reduction_pct:
                    st.metric("Variance reduced", f"{cuped.variance_reduction_pct:.1f}%",
                              delta=f"-{cuped.variance_reduction_pct:.0f}% variance")

    # ── Tab 3: Multiple testing ───────────────────────────────────────────────
    with tab3:
        st.subheader("Multiple Testing Correction")
        st.info("When testing multiple metrics, your false positive rate increases. "
                "Benjamini-Hochberg (BH) is recommended — it controls the False Discovery Rate.")

        n_tests = st.slider("Number of metrics tested", 2, 10, 5)
        alpha_multi = st.select_slider("Family-wise α", [0.01, 0.05, 0.10], value=0.05)
        method_multi = st.radio("Correction method", ["bh", "bonferroni"],
                                format_func=lambda x: "Benjamini-Hochberg (FDR)" if x == "bh" else "Bonferroni (FWER)",
                                horizontal=True)

        st.markdown("**Enter raw p-values:**")
        p_vals = []
        metric_names = []
        cols_row = st.columns(min(n_tests, 5))
        for i in range(n_tests):
            col_idx = i % 5
            with cols_row[col_idx]:
                name = st.text_input(f"Metric {i+1} name", value=f"metric_{i+1}", key=f"mname_{i}")
                pv = st.number_input(f"p-value", 0.0, 1.0, [0.01, 0.04, 0.20, 0.03, 0.15, 0.45, 0.02, 0.30, 0.08, 0.50][i],
                                     step=0.001, format="%.4f", key=f"pv_{i}")
                p_vals.append(pv)
                metric_names.append(name)

        corrected = correct_multiple_tests(p_vals, alpha=alpha_multi, method=method_multi)
        corrected.insert(0, "metric", metric_names)

        def highlight_sig(row):
            if row["significant"]:
                return ["background-color: #d4edda"] * len(row)
            return [""] * len(row)

        st.dataframe(corrected.style.apply(highlight_sig, axis=1), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: BAYESIAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Bayesian Analysis":
    st.title("🎯 Bayesian A/B Testing")
    st.markdown(
        "Bayesian testing gives you **probability that treatment is better** "
        "and **expected loss** — more intuitive than p-values for stakeholders."
    )

    col1, col2 = st.columns(2)
    with col1:
        b_ctrl_n = st.number_input("Control users", 100, 1000000, 5000, key="b_cn")
        b_ctrl_c = st.number_input("Control conversions", 0, int(b_ctrl_n), 500, key="b_cc")
        prior_a = st.number_input("Prior α (Beta)", 0.1, 100.0, 1.0, step=0.5,
                                   help="Beta(1,1) = flat/uniform prior")
        prior_b = st.number_input("Prior β (Beta)", 0.1, 100.0, 1.0, step=0.5)

    with col2:
        b_trt_n = st.number_input("Treatment users", 100, 1000000, 5000, key="b_tn")
        b_trt_c = st.number_input("Treatment conversions", 0, int(b_trt_n), 560, key="b_tc")
        loss_thresh = st.number_input("Decision loss threshold", 0.0001, 0.05, 0.001,
                                       format="%.4f",
                                       help="Declare winner when E[loss] < this value")
        ci_level = st.select_slider("Credible interval level", [0.90, 0.95, 0.99], value=0.95)

    result = bayesian_proportion_test(
        int(b_ctrl_c), int(b_ctrl_n),
        int(b_trt_c), int(b_trt_n),
        prior_alpha=prior_a, prior_beta=prior_b,
        loss_threshold=loss_thresh, ci_level=ci_level,
    )

    col3, col4, col5, col6 = st.columns(4)
    col3.metric("P(treatment > control)", f"{result.prob_treatment_better:.1%}")
    col4.metric("P(control > treatment)", f"{result.prob_control_better:.1%}")
    col5.metric("E[loss | choose trt]", f"{result.expected_loss_treatment:.4f}")
    col6.metric("Credible interval",
                f"[{result.credible_interval_lower:.3f}, {result.credible_interval_upper:.3f}]")

    st.markdown(f"### Decision: {result.decision}")

    # Posterior visualisation
    rng3 = np.random.default_rng(42)
    n_samples = 50000
    samples_ctrl = rng3.beta(prior_a + b_ctrl_c, prior_b + (b_ctrl_n - b_ctrl_c), n_samples)
    samples_trt = rng3.beta(prior_a + b_trt_c, prior_b + (b_trt_n - b_trt_c), n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Posterior distributions
    ax1 = axes[0]
    x_range = np.linspace(
        min(samples_ctrl.min(), samples_trt.min()) * 0.9,
        max(samples_ctrl.max(), samples_trt.max()) * 1.1,
        500,
    )
    from scipy.stats import gaussian_kde
    kde_ctrl = gaussian_kde(samples_ctrl)
    kde_trt = gaussian_kde(samples_trt)
    ax1.fill_between(x_range, kde_ctrl(x_range), alpha=0.5, color="#3498db", label="Control")
    ax1.fill_between(x_range, kde_trt(x_range), alpha=0.5, color="#e74c3c", label="Treatment")
    ax1.set_xlabel("Conversion Rate")
    ax1.set_ylabel("Posterior Density")
    ax1.set_title("Posterior Distributions")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Distribution of lift
    ax2 = axes[1]
    diff_samples = samples_trt - samples_ctrl
    ax2.hist(diff_samples, bins=80, color="#9b59b6", alpha=0.75, density=True)
    ax2.axvline(0, color="black", lw=2, ls="--", label="No effect")
    ax2.axvline(diff_samples.mean(), color="#e74c3c", lw=2, label=f"Mean lift={diff_samples.mean():.3f}")
    # Shade CI
    lo, hi = result.credible_interval_lower, result.credible_interval_upper
    x_ci = np.linspace(lo, hi, 200)
    kde_diff = gaussian_kde(diff_samples)
    ax2.fill_between(x_ci, kde_diff(x_ci), alpha=0.3, color="#e74c3c",
                     label=f"{int(ci_level*100)}% CI")
    ax2.set_xlabel("Treatment − Control")
    ax2.set_title(f"Posterior Lift Distribution\nP(B>A) = {result.prob_treatment_better:.1%}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📖 Bayesian vs Frequentist — When to use which?"):
        st.markdown("""
| | **Frequentist** | **Bayesian** |
|---|---|---|
| Interpretable as... | "If there's no effect, how unlikely is this data?" | "Given this data, how probable is the treatment better?" |
| Requires fixed n? | ✅ Yes (peeking inflates α) | ❌ No (valid at any point with mSPRT) |
| Outputs | p-value, CI | P(B>A), Expected loss, Credible interval |
| Best for... | Formal hypothesis tests, regulated decisions | Business decisions, real-time dashboards |
| Stakeholder friendliness | Low (p-values are misunderstood) | High (probability % is intuitive) |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: SEQUENTIAL & NOVELTY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⏱ Sequential & Novelty":
    st.title("⏱ Sequential Testing & Novelty Effect Detection")

    tab1, tab2 = st.tabs(["SPRT Sequential Monitor", "Novelty Effect Detector"])

    with tab1:
        st.subheader("Sequential Probability Ratio Test (SPRT)")
        st.info("SPRT lets you stop experiments early when evidence is strong — "
                "without inflating your false positive rate.")

        col1, col2 = st.columns(2)
        with col1:
            sprt_p0 = st.slider("Null rate (p₀)", 0.01, 0.50, 0.10, 0.005)
            sprt_mde = st.slider("MDE (absolute)", 0.001, 0.10, 0.02, 0.001)
            sprt_alpha = st.select_slider("α", [0.01, 0.05, 0.10], value=0.05)
            sprt_beta = st.select_slider("β (1 - power)", [0.05, 0.10, 0.15, 0.20], value=0.20)
        with col2:
            true_effect = st.slider("True effect (simulation)", 0.0, 0.10, 0.025, 0.001,
                                     help="True treatment effect for simulation")
            n_sim = st.slider("Max users to simulate", 1000, 20000, 8000, 500)
            seed = st.number_input("Random seed", value=42, step=1)

        if st.button("▶ Run SPRT Simulation"):
            rng4 = np.random.default_rng(int(seed))
            ctrl_obs = rng4.binomial(1, sprt_p0, n_sim)
            trt_obs = rng4.binomial(1, sprt_p0 + true_effect, n_sim)

            monitor = SPRTMonitor(p0=sprt_p0, p1=sprt_p0 + sprt_mde,
                                  alpha=sprt_alpha, beta=sprt_beta)
            monitor.run_batch(ctrl_obs, trt_obs)
            hist = monitor.history_df()

            # Find decision point
            decision_rows = hist[hist["decision"] != "continue"]
            stop_n = int(decision_rows.iloc[0]["n"]) if not decision_rows.empty else n_sim

            col3, col4, col5 = st.columns(3)
            col3.metric("Stopped at n", f"{stop_n:,}/arm")
            col4.metric("Sample savings", f"{(n_sim - stop_n)/n_sim:.0%}",
                        delta=f"{n_sim - stop_n:,} fewer users")
            decision_val = hist.iloc[-1]["decision"] if not decision_rows.empty else "continue"
            col5.metric("Decision", decision_val.replace("_", " ").upper())

            # Plot LLR trajectory
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hist["n"], hist["llr"], color="#2c3e50", lw=1.5, label="Log-Likelihood Ratio")
            ax.axhline(monitor.upper, color="#27ae60", ls="--", lw=2, label=f"Accept H₁ ({monitor.upper:.2f})")
            ax.axhline(monitor.lower, color="#e74c3c", ls="--", lw=2, label=f"Accept H₀ ({monitor.lower:.2f})")
            if not decision_rows.empty:
                ax.axvline(stop_n, color="orange", ls=":", lw=2, label=f"Stopped at n={stop_n:,}")
            ax.fill_between(hist["n"], monitor.lower, monitor.upper, alpha=0.05, color="gray")
            ax.set_xlabel("Users observed per arm")
            ax.set_ylabel("Log-Likelihood Ratio (LLR)")
            ax.set_title("SPRT Monitoring Chart")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.subheader("Novelty (Primacy) Effect Detector")
        st.info("A novelty effect is when users are initially excited by a change, "
                "but their behaviour returns to baseline over time. "
                "This can make a neutral experiment look falsely positive.")

        col1, col2 = st.columns(2)
        with col1:
            n_days = st.slider("Experiment duration (days)", 14, 56, 28, 7)
            base_cr = st.slider("Baseline conversion rate", 0.05, 0.30, 0.10, 0.01)
            early_effect = st.slider("Early lift (week 1)", 0.0, 0.10, 0.05, 0.005)
        with col2:
            late_effect = st.slider("Late lift (final week)", 0.0, 0.10, 0.02, 0.005)
            noise = st.slider("Daily noise (σ)", 0.001, 0.02, 0.005, 0.001)
            n_windows_novelty = st.select_slider("Windows", [2, 3, 4, 6], value=4)

        # Generate simulated time series
        rng5 = np.random.default_rng(55)
        dates = pd.date_range("2024-01-01", periods=n_days)

        ctrl_daily = base_cr + rng5.normal(0, noise, n_days)

        # Interpolate effect from early to late linearly
        effect_curve = np.linspace(early_effect, late_effect, n_days)
        trt_daily = ctrl_daily + effect_curve + rng5.normal(0, noise, n_days)

        ctrl_ts = pd.DataFrame({"date": dates, "conversion_rate": np.clip(ctrl_daily, 0, 1)})
        trt_ts = pd.DataFrame({"date": dates, "conversion_rate": np.clip(trt_daily, 0, 1)})

        novelty = detect_novelty_effect(ctrl_ts, trt_ts, n_windows=n_windows_novelty)

        st.markdown(f"### {'⚠️ Novelty Effect Detected' if novelty.has_novelty_effect else '✅ No Novelty Effect'}")
        col3, col4, col5 = st.columns(3)
        col3.metric("Early lift", f"{novelty.early_lift:+.4f}")
        col4.metric("Late lift", f"{novelty.late_lift:+.4f}")
        col5.metric("Lift decay", f"{novelty.lift_decay_pct:.1f}%",
                    delta=f"{'⚠️ ' if novelty.has_novelty_effect else ''}{novelty.lift_decay_pct:.0f}%",
                    delta_color="inverse" if novelty.has_novelty_effect else "normal")

        st.info(f"**Recommendation:** {novelty.recommendation}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax1 = axes[0]
        ax1.plot(dates, ctrl_ts["conversion_rate"], color="#3498db", lw=1.5, label="Control")
        ax1.plot(dates, trt_ts["conversion_rate"], color="#e74c3c", lw=1.5, label="Treatment")
        ax1.set_title("Daily Conversion Rate")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Conversion Rate")
        ax1.legend()
        ax1.grid(alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

        ax2 = axes[1]
        ax2.bar(novelty.window_labels, novelty.window_lifts,
                color=["#e74c3c" if l > 0 else "#3498db" for l in novelty.window_lifts],
                alpha=0.8)
        ax2.axhline(0, color="black", lw=1)
        ax2.set_title("Treatment Lift by Time Window")
        ax2.set_ylabel("Lift (treatment − control)")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
