"""
Adapters that wrap recsys-adtech models behind the ModelAdapter protocol.

Import strategy: recsys-adtech uses implicit relative imports (e.g. `from features
import FEATURE_DIM` inside 05-adtech/src/). We resolve these by temporarily
adding the right source directories to sys.path before importing.
"""
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Path resolution ────────────────────────────────────────────────────────────
_RECSYS_ROOT = Path(__file__).parents[3] / "recsys-adtech"
_RANKING_SRC = _RECSYS_ROOT / "02-ranking" / "src"
_ADTECH_SRC = _RECSYS_ROOT / "05-adtech" / "src"
_SHARED = _RECSYS_ROOT / "shared"

for _p in (_RECSYS_ROOT, _RANKING_SRC, _ADTECH_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ── PointwiseScorerAdapter ─────────────────────────────────────────────────────

class PointwiseScorerAdapter:
    """
    Wraps recsys-adtech PointwiseScorer (LightGBM) behind ModelAdapter.

    PointwiseScorer.fit takes np.ndarray; ModelAdapter.fit takes pd.DataFrame.
    PointwiseScorer has no save/load — we persist via lgb.Booster.save_model().
    """

    def __init__(self, n_rounds: int = 100, **lgbm_params) -> None:
        self._n_rounds = n_rounds
        self._lgbm_params = lgbm_params
        self._scorer: Any = None

    def _build_scorer(self) -> Any:
        from pointwise import PointwiseScorer  # resolved via sys.path
        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "verbosity": -1,
            "seed": 42,
        }
        params.update(self._lgbm_params)
        return PointwiseScorer(params=params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._scorer = self._build_scorer()
        self._scorer.fit(X.values.astype(np.float32), y.values.astype(np.float32), n_rounds=self._n_rounds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._scorer is None or self._scorer._model is None:
            return np.full(len(X), 0.5)
        return self._scorer.predict(X.values.astype(np.float32))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # LightGBM Booster has its own text serialization — more portable than pickle
        self._scorer._model.save_model(path + ".lgbm")
        # Also save metadata
        with open(path, "wb") as f:
            pickle.dump({"n_rounds": self._n_rounds, "params": self._lgbm_params}, f)

    def load(self, path: str) -> None:
        import lightgbm as lgb
        with open(path, "rb") as f:
            meta = pickle.load(f)
        self._n_rounds = meta["n_rounds"]
        self._lgbm_params = meta["params"]
        self._scorer = self._build_scorer()
        self._scorer._model = lgb.Booster(model_file=path + ".lgbm")


# ── DeepFMCTRAdapter ───────────────────────────────────────────────────────────

class DeepFMCTRAdapter:
    """
    Wraps recsys-adtech DeepFM (CTR model) behind ModelAdapter.

    DeepFM is a PyTorch nn.Module; train_deepfm() handles the training loop.
    save/load via torch.save on the state_dict — avoids pickle of nn.Module.
    """

    def __init__(self, n_epochs: int = 10, batch_size: int = 512, lr: float = 1e-3, seed: int = 42) -> None:
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._lr = lr
        self._seed = seed
        self._model: Any = None
        self._in_dim: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import torch
        from ctr_model import train_deepfm  # resolved via sys.path
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)
        self._in_dim = X_np.shape[1]
        # train_deepfm returns (model, val_scores, val_labels) since the
        # recsys-adtech update that added calibration support.
        result = train_deepfm(
            X_np, y_np,
            n_epochs=self._n_epochs,
            batch_size=self._batch_size,
            lr=self._lr,
            seed=self._seed,
        )
        self._model = result[0] if isinstance(result, tuple) else result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.full(len(X), 0.5)
        import torch
        X_t = torch.tensor(X.values.astype(np.float32))
        self._model.eval()
        with torch.no_grad():
            scores = self._model(X_t).numpy()
        return scores

    def save(self, path: str) -> None:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._model.state_dict(),
            "in_dim": self._in_dim,
            "config": {
                "n_epochs": self._n_epochs,
                "batch_size": self._batch_size,
                "lr": self._lr,
                "seed": self._seed,
            },
        }, path)

    def load(self, path: str) -> None:
        import torch
        from ctr_model import DeepFM  # resolved via sys.path
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        self._in_dim = checkpoint["in_dim"]
        self._n_epochs = cfg["n_epochs"]
        self._batch_size = cfg["batch_size"]
        self._lr = cfg["lr"]
        self._seed = cfg["seed"]
        self._model = DeepFM(in_dim=self._in_dim)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()
