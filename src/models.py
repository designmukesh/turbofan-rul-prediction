"""Model definitions for RUL regression.

Baseline first (Ridge — linear, interpretable, cheap), then two tree
ensembles. No deep learning: with ~17k window samples and 84 tabular
features, gradient boosting is the right tool — LSTMs add complexity
without a demonstrated gain on this framing (documented as future work).
"""
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


def get_models(random_state=42):
    """Return the compared model set. Same configs as notebook 02_modeling."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=random_state
        ),
    }


WINNER = "HistGradientBoosting"
