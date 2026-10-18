"""Turbofan RUL — maintenance decision demo.

Decision-oriented view of the RUL model: pick a test engine, see its sensor
traces, the model's RUL estimate across its life, a risk state, and an
illustrative recommended action.

Honesty labels: C-MAPSS is *simulated* data; risk thresholds and action rules
are illustrative, not validated maintenance policy; true RUL is shown only
because this is a demo (in reality it is hidden).
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_subset, RUL_CAP  # noqa: E402
from src.features import window_features  # noqa: E402

st.set_page_config(page_title="Turbofan RUL — Decision Demo", layout="wide")
st.title("Turbofan Engine RUL — Maintenance Decision Demo")
st.caption(
    "NASA C-MAPSS FD001 (simulated data) · HistGradientBoosting on 30-cycle "
    "window statistics · Demo only — not a certified maintenance product."
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "models" / "hgb_rul.joblib")
    card = pd.read_json(ROOT / "models" / "model_card.json", typ="series")
    train, test, rul = load_subset("FD001", ROOT / "data")
    return model, card, test, rul


model, card, test, rul_truth = load_artifacts()

# ---- sidebar ----
engines = sorted(test["unit"].unique())
unit = st.sidebar.selectbox("Test engine", engines, index=0)
true_rul = float(rul_truth["rul"].iloc[unit - 1])
eng = test[test["unit"] == unit].sort_values("cycle").reset_index(drop=True)

Xw, yw, _ = window_features(eng)
pred_w = model.predict(Xw)
cycles_w = np.arange(30, len(eng) + 1)
final_pred = float(pred_w[-1])

# ---- risk state (illustrative thresholds, labeled as such) ----
if final_pred < 30:
    risk, color, action = "HIGH", "red", (
        "Schedule boroscope inspection at the next opportunity; "
        "plan shop visit — do not extend past 30 predicted cycles."
    )
elif final_pred < 60:
    risk, color, action = "MEDIUM", "orange", (
        "Increase monitoring cadence; tentative shop-visit slot in planning."
    )
else:
    risk, color, action = "LOW", "green", (
        "Continue normal operation; re-score after next flights."
    )

# ---- KPI row ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted RUL", f"{final_pred:.0f} cycles")
c2.metric("True RUL (demo only)", f"{true_rul:.0f} cycles",
          delta=f"{final_pred - true_rul:+.0f} err")
c3.metric(f"Risk: :{color}[{risk}]", "")
c4.metric("Test-set RMSE", f"{float(card['test_metrics']['rmse']):.1f} cycles")
st.info("**Recommended action (illustrative rules, not validated policy):** " + action)

# ---- charts ----
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for s, lbl in [("s3", "T30 HPC outlet temp"), ("s4", "T50 LPT outlet temp")]:
    axes[0].plot(eng["cycle"], eng[s], label=lbl, lw=1.2)
axes[0].axvline(len(eng), color="k", ls="--", lw=0.8)
axes[0].set_ylabel("sensor value")
axes[0].set_title(f"Engine {unit}: key sensor traces (dashed = data ends / 'today')")
axes[0].legend(fontsize=9)

axes[1].plot(cycles_w, yw, label="true RUL", color="k", lw=1.5)
axes[1].plot(cycles_w, pred_w, label="predicted RUL", color="steelblue", lw=1.2)
axes[1].axhline(RUL_CAP, color="gray", ls=":", lw=0.8, label="RUL cap (125)")
axes[1].set_xlabel("cycle")
axes[1].set_ylabel("RUL (cycles)")
axes[1].set_title("Model tracking degradation over the engine's life")
axes[1].legend(fontsize=9)
fig.tight_layout()
st.pyplot(fig)

with st.expander("How to read this (and what it does not prove)"):
    st.markdown(
        "- **Predicted vs true RUL** shows whether the model tracks degradation; "
        "early life is flat because the 125-cycle cap says 'healthy' — sensors show "
        "no trend there.\n"
        "- **Risk + action** come from fixed thresholds on predicted RUL. They are "
        "placeholder rules to show the *decision flow*, not airline policy.\n"
        "- This is **simulated** data (FD001: one operating condition, one fault mode). "
        "Field accuracy, savings, and certification are *not* demonstrated here."
    )
