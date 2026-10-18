"""Train the final RUL model artifact for the Streamlit demo.

Trains HistGradientBoosting (the validated winner) on ALL 100 train engines
and saves it with its feature schema to models/. The test set is scored once
here for the artifact's model card — this mirrors notebook 02's final
evaluation and does not feed back into any decision.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "..")
import joblib
import numpy as np

from src.data import load_subset
from src.features import window_features, last_window_per_engine
from src.models import get_models, WINNER
from src.evaluation import evaluate

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

train, test, rul_test = load_subset("FD001", ROOT / "data")
t0 = time.time()
X, y, _ = window_features(train)
print(f"training windows: {len(X):,} | features: {X.shape[1]} | built in {time.time()-t0:.0f}s")

model = get_models()["HistGradientBoosting"]
t0 = time.time()
model.fit(X, y)
print(f"trained {WINNER} in {time.time()-t0:.0f}s")

# one final, honest scoring on the untouched test set (model card only)
Xte, engines = last_window_per_engine(test)
pred = model.predict(Xte)
m = evaluate(rul_test["rul"].to_numpy(), pred)
print(f"test: RMSE {m['rmse']:.2f} | PHM {m['phm_score']:.0f} | MAE {m['mae']:.2f}")

joblib.dump(model, MODEL_DIR / "hgb_rul.joblib")
joblib.dump(list(X.columns), MODEL_DIR / "feature_names.joblib")
with open(MODEL_DIR / "model_card.json", "w") as f:
    json.dump({
        "model": WINNER,
        "trained_on": "FD001, all 100 train engines, 30-cycle windows, 84 features",
        "test_metrics": m,
        "rul_cap": 125,
        "warning": "Simulated C-MAPSS data. Demo artifact, not a certified product.",
    }, f, indent=2)
print("saved models/hgb_rul.joblib + model_card.json")
