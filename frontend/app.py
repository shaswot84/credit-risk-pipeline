"""
Credit Risk Prediction — Streamlit frontend.

Left: input form + speedometer result.
Right: confusion matrix, evaluation metrics, feature importance, scenarios.
"""

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Credit Risk", page_icon="💳", layout="wide")
API = "http://localhost:8000"

PAY_LABELS = {
    -2: "No credit used", -1: "Paid duly", 0: "Revolving credit",
    1: "1m late", 2: "2m late", 3: "3m late", 4: "4m late",
    5: "5m late", 6: "6m late", 7: "7m late", 8: "8m late",
}

PAY_TOOLTIPS = {
    "PAY_SEP": -1, "PAY_AUG": -1, "PAY_JUL": -1, "PAY_JUN": -1, "PAY_MAY": -1, "PAY_APR": -1,
}

# ---------------------------------------------------------------------------
# Pre-fetch static model info
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_model_info(api_version=1):
    try:
        r = requests.get(f"{API}/model-info", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None

INFO = fetch_model_info(api_version=2)

# ---------------------------------------------------------------------------
# Speedometer
# ---------------------------------------------------------------------------
def speedometer(prob, threshold):
    warning_z = threshold * 0.6
    color = "#2ecc71" if prob < warning_z else "#f1c40f" if prob < threshold else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"font": {"size": 34, "color": color}, "valueformat": ".1%"},
        title={"text": "Default Probability", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 1], "tickformat": ".0%", "nticks": 6},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0, warning_z], "color": "#d5f5e3"},
                {"range": [warning_z, threshold], "color": "#fef9e7"},
                {"range": [threshold, 1], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.6, "value": threshold,
            },
        },
    ))
    fig.update_layout(
        height=270, margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "gray"},
    )
    return fig

# ---------------------------------------------------------------------------
# Feature tooltips
# ---------------------------------------------------------------------------
FEATURE_HELP = {
    "LIMIT_BAL": "Credit limit in NTD (10,000 – 1,000,000). Higher limits → lower default risk.",
    "BILL_SEP": "Bill amount for September in NTD. How much was owed.",
    "PAYMENT_SEP": "Payment amount for September in NTD. How much was actually paid.",
    "PAY_SEP": "Repayment status for September. -1=paid duly, 0=revolving, 1+=months late. Most predictive feature.",
    "PAY_AUG": "Repayment status for August.",
    "PAY_JUL": "Repayment status for July.",
    "PAY_JUN": "Repayment status for June.",
    "PAY_MAY": "Repayment status for May.",
    "PAY_APR": "Repayment status for April (oldest).",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    THRESHOLD = st.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.01,
                          help="If probability ≥ threshold → classified as Default.")
    MODEL = st.selectbox(
        "Model", ["ensemble", "vanilla"],
        format_func=lambda x: "🎯 Ensemble (4 models)" if x == "ensemble" else "🧠 Vanilla MLP",
    )
    st.markdown("---")
    st.markdown("##### Repayment Status Key")
    for v in [0, -1, -2, 1, 2, 3]:
        i = "🟢" if v <= 0 else "🟡" if v <= 2 else "🔴"
        st.caption(f"{i} **{v:+d}** — {PAY_LABELS[v]}")

# ---------------------------------------------------------------------------
# Main: two columns
# ---------------------------------------------------------------------------
left, right = st.columns([1.15, 1])

# ===== LEFT =====
with left:
    st.title("💳 Credit Risk")
    st.caption("Ensemble: XGBoost + RF + MLP-A + MLP-B  |  FocalLoss MLP baseline  |  30K UCI clients")

    c1, c2 = st.columns(2)
    with c1:
        limit_bal = st.number_input(
            "Credit Limit (NTD)", 10000, 2000000, 50000, 10000, format="%d",
            help=FEATURE_HELP["LIMIT_BAL"],
        )
        bill_sep = st.number_input(
            "Latest Bill (NTD)", -200000, 2000000, 25000, 1000, format="%d",
            help=FEATURE_HELP["BILL_SEP"],
        )
        payment_sep = st.number_input(
            "Latest Payment (NTD)", 0, 2000000, 2000, 500, format="%d",
            help=FEATURE_HELP["PAYMENT_SEP"],
        )
    with c2:
        st.caption("Repayment History (Sep → Apr)")
        months = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
        keys = ["PAY_SEP", "PAY_AUG", "PAY_JUL", "PAY_JUN", "PAY_MAY", "PAY_APR"]
        pay_vals = {}
        for label, key in zip(months, keys):
            pay_vals[key] = st.selectbox(
                label, range(-2, 9), index=2,
                format_func=lambda x: f"{x:+d} {PAY_LABELS[x]}",
                help=FEATURE_HELP[key],
                key=key,
            )

    pay_list = list(pay_vals.values())
    delay = sum(max(0, v) for v in pay_list) * 30
    ratio = payment_sep / (abs(bill_sep) + 1)

    # --- Predict ---
    st.markdown("---")
    _, col_b, _ = st.columns([0.3, 0.4, 0.3])
    with col_b:
        clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

    if clicked:
        payload = {
            "LIMIT_BAL": float(limit_bal), "BILL_SEP": float(bill_sep),
            "PAYMENT_SEP": float(payment_sep), **{k: int(v) for k, v in pay_vals.items()},
        }
        with st.spinner(""):
            try:
                r = requests.post(f"{API}/predict", json=payload,
                                  params={"threshold": THRESHOLD, "model": MODEL}, timeout=10)
                r.raise_for_status()
                res = r.json()
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()

        prob, pred = res["probability"], res["prediction"]

        st.plotly_chart(speedometer(prob, THRESHOLD), use_container_width=True)

        if prob >= THRESHOLD:
            st.error(f"### 🔴 {pred} — HIGH RISK")
        else:
            st.success(f"### 🟢 {pred} — LOW RISK")

        st.caption(f"{res['model_used']}  |  {delay}d delay  |  Pay/Bill: **{ratio:.2f}**")

        if res.get("models"):
            st.markdown("**Model Votes**")
            st.caption("Per-model probability breakdown (ensemble only)")
            dfm = pd.DataFrame(
                {"Model": list(res["models"].keys()), "Prob": list(res["models"].values())}
            ).sort_values("Prob")
            st.bar_chart(dfm.set_index("Model"), use_container_width=True, height=180)

# ===== RIGHT =====
with right:
    if not INFO:
        st.warning("Start the backend API to see model performance.")
        st.stop()

    # --- Eval metrics ---
    st.markdown("## 📊 Model Performance")
    em = INFO["ensemble_metrics"]
    vm = INFO["vanilla_metrics"]
    metrics_df = pd.DataFrame(
        {"AUC-ROC": [em["auc"], vm["auc"]],
         "Avg Precision": [em["ap"], vm["ap"]],
         "Brier": [em["brier"], vm["brier"]],
         "Precision": [em["precision"], vm["precision"]],
         "Recall": [em["recall"], vm["recall"]],
         "F1": [em["f1"], vm["f1"]]},
        index=["Ensemble (Avg)", "Vanilla MLP"],
    )
    st.dataframe(
        metrics_df.style.format("{:.4f}"),
        column_config={
            "AUC-ROC": st.column_config.NumberColumn(
                "AUC-ROC",
                help="Area Under the ROC Curve — how well the model ranks defaulters above non-defaulters (0.5=random, 1.0=perfect).",
            ),
            "Avg Precision": st.column_config.NumberColumn(
                "Avg Precision",
                help="Average Precision (PR AUC) — measures precision across all recall thresholds; more informative than ROC for imbalanced classes.",
            ),
            "Brier": st.column_config.NumberColumn(
                "Brier",
                help="Brier Score — mean squared error of predicted probabilities (0.0=perfect, ~0.25=naive prediction of 22% base rate). Lower is better.",
            ),
            "Precision": st.column_config.NumberColumn(
                "Precision",
                help="Precision = TP / (TP + FP) — of all predicted defaults, how many were correct. High precision means fewer false alarms.",
            ),
            "Recall": st.column_config.NumberColumn(
                "Recall",
                help="Recall = TP / (TP + FN) — of all actual defaulters, how many were caught. High recall means fewer missed defaulters.",
            ),
            "F1": st.column_config.NumberColumn(
                "F1",
                help="F1 Score — harmonic mean of Precision and Recall (best viewed alongside its components to understand the trade-off).",
            ),
        },
        use_container_width=True,
    )
    st.caption(f"On {INFO['test_set_size']:,} hold-out samples  ·  threshold = 0.5")

    # --- Confusion matrix ---
    st.divider()
    st.markdown("### Confusion Matrix")
    if MODEL == "ensemble":
        cm = INFO["ensemble_confusion_matrix"]
        cm_label = "Ensemble"
    else:
        cm = INFO["vanilla_confusion_matrix"]
        cm_label = "Vanilla MLP"
    cml = cm["labels"]
    cm_df = pd.DataFrame(
        cm["matrix"],
        index=[f"Actual {l}" for l in cml],
        columns=[f"Predicted {l}" for l in cml],
    )
    st.dataframe(cm_df, use_container_width=True)
    tpr = cm["tp"] / (cm["tp"] + cm["fn"])
    ppv = cm["tp"] / (cm["tp"] + cm["fp"])
    st.caption(
        f"{cm_label}  ·  TN {cm['tn']:,}  FP {cm['fp']:,}  "
        f"FN {cm['fn']:,}  TP {cm['tp']:,}  "
        f"·  Recall {tpr:.1%}  ·  Precision {ppv:.1%}"
    )

    # --- Scenarios ---
    st.divider()
    st.markdown("### Scenario Walkthrough")
    st.caption("Real-world examples showing each confusion-matrix quadrant")
    for s in INFO["scenarios"]:
        icon = {"True Positive": "✅", "True Negative": "✅", "False Positive": "⚠️", "False Negative": "⚠️"}
        with st.expander(f"{icon.get(s['type'],'')} {s['type']} — {s['description']}"):
            for ex in s["examples"]:
                bar = "🟢" if ex["actual"] == ex["predicted"] else "🔴"
                st.caption(
                    f"{bar} **{ex['prob']:.0%}**  — {ex['scenario']}  "
                    f"(actual: {ex['actual']}, predicted: {ex['predicted']})"
                )

    # --- Feature importance ---
    st.divider()
    st.markdown("### 📈 Feature Importance")
    if MODEL == "ensemble":
        fi = INFO["ensemble_feature_importance"]
        st.caption("RandomForest · Mean Decrease in Impurity")
    else:
        fi = INFO["vanilla_feature_importance"]
        st.caption("Vanilla MLP · Permutation Importance (10 repeats)")
    df_fi = pd.DataFrame(fi).set_index("feature")
    st.bar_chart(df_fi, use_container_width=True, height=260)
    st.dataframe(df_fi.style.format("{:.3f}"), use_container_width=True)
