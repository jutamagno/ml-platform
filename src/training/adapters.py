import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ModelAdapter(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


class LightGBMAdapter:
    def __init__(self, **lgbm_params) -> None:
        import lightgbm as lgb
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_estimators": 100,
        }
        params.update(lgbm_params)
        self._model = lgb.LGBMClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


class PyTorchAdapter:
    """Wraps any nn.Module with a sklearn-like interface."""

    def __init__(self, module_factory, input_dim: int, lr: float = 1e-3, epochs: int = 10) -> None:
        import torch
        self._factory = module_factory
        self._input_dim = input_dim
        self._lr = lr
        self._epochs = epochs
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        self._model = self._factory(self._input_dim)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.BCEWithLogitsLoss()

        self._model.train()
        for _ in range(self._epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(xb).squeeze(), yb)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X.values, dtype=torch.float32)
            logits = self._model(X_t).squeeze()
            return torch.sigmoid(logits).numpy()

    def save(self, path: str) -> None:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self._model.state_dict(), "factory": self._factory, "input_dim": self._input_dim}, path)

    def load(self, path: str) -> None:
        import torch
        checkpoint = torch.load(path, map_location="cpu")
        self._model = checkpoint["factory"](checkpoint["input_dim"])
        self._model.load_state_dict(checkpoint["model_state"])
