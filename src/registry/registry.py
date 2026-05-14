import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.training.adapters import ModelAdapter

logger = logging.getLogger(__name__)

REGISTRY_ROOT_DEFAULT = "/tmp/ml-platform/registry"


@dataclass
class ModelVersion:
    model_name: str
    version: str
    stage: str
    metrics: dict[str, float]
    meta: dict
    registered_at: datetime
    metrics_locked: bool = False


class ModelRegistry:
    """
    Filesystem-backed, immutable version store.
    Versions are locked once registered; metrics can be updated once
    (to add online metrics post-deployment), then locked.
    """

    def __init__(self, root: str = REGISTRY_ROOT_DEFAULT) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, dict[str, ModelVersion]] = {}  # model_name → version → ModelVersion

    def register(
        self,
        model_name: str,
        adapter: ModelAdapter,
        metrics: dict[str, float],
        meta: dict,
    ) -> str:
        version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")
        version_dir = self._root / model_name / version
        if version_dir.exists():
            raise ValueError(f"Version {version} already registered for {model_name}")

        model_dir = self._root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(dir=model_dir))
        try:
            adapter.save(str(tmp_dir / "model.pkl"))
            (tmp_dir / "training_meta.json").write_text(json.dumps(meta, default=str))
            (tmp_dir / "metrics.json").write_text(json.dumps(metrics, default=str))
            (tmp_dir / "stage.txt").write_text("staging")
            tmp_dir.rename(version_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            stage="staging",
            metrics=metrics,
            meta=meta,
            registered_at=datetime.now(timezone.utc),
        )
        self._versions.setdefault(model_name, {})[version] = model_version
        logger.info("Registered %s v%s | metrics=%s", model_name, version, metrics)
        return version

    def get_version(self, model_name: str, version: str) -> ModelVersion:
        self._ensure_loaded(model_name, version)
        return self._versions[model_name][version]

    def get_latest(self, model_name: str, stage: str = "production") -> ModelVersion:
        versions = self.list_versions(model_name, stage=stage)
        if not versions:
            raise KeyError(f"No {stage} version for model {model_name}")
        return versions[-1]

    def list_versions(self, model_name: str, stage: Optional[str] = None) -> list[ModelVersion]:
        model_dir = self._root / model_name
        if not model_dir.exists():
            return []
        for version_dir in sorted(model_dir.iterdir()):
            version = version_dir.name
            if version not in self._versions.get(model_name, {}):
                self._load_version(model_name, version)
        all_versions = list(self._versions.get(model_name, {}).values())
        all_versions.sort(key=lambda v: v.registered_at)
        if stage is not None:
            all_versions = [v for v in all_versions if v.stage == stage]
        return all_versions

    def promote(self, model_name: str, version: str, stage: str) -> None:
        mv = self.get_version(model_name, version)
        mv.stage = stage
        stage_file = self._root / model_name / version / "stage.txt"
        stage_file.write_text(stage)

    def update_metrics(self, model_name: str, version: str, online_metrics: dict[str, float]) -> None:
        mv = self.get_version(model_name, version)
        if mv.metrics_locked:
            raise ValueError(f"Metrics for {model_name} v{version} are locked")
        mv.metrics.update(online_metrics)
        mv.metrics_locked = True
        version_dir = self._root / model_name / version
        (version_dir / "metrics.json").write_text(json.dumps(mv.metrics, default=str))
        (version_dir / "metrics_locked.txt").write_text("1")

    def compare(self, model_name: str, version_a: str, version_b: str) -> pd.DataFrame:
        va = self.get_version(model_name, version_a)
        vb = self.get_version(model_name, version_b)
        all_keys = set(va.metrics) | set(vb.metrics)
        rows = []
        for key in sorted(all_keys):
            rows.append({
                "metric": key,
                version_a: va.metrics.get(key),
                version_b: vb.metrics.get(key),
                "diff": (vb.metrics.get(key, 0) or 0) - (va.metrics.get(key, 0) or 0),
            })
        return pd.DataFrame(rows)

    def _ensure_loaded(self, model_name: str, version: str) -> None:
        if model_name not in self._versions or version not in self._versions[model_name]:
            self._load_version(model_name, version)

    def _load_version(self, model_name: str, version: str) -> None:
        version_dir = self._root / model_name / version
        if not version_dir.exists():
            raise KeyError(f"Version {version} not found for model {model_name}")
        metrics = json.loads((version_dir / "metrics.json").read_text())
        meta = json.loads((version_dir / "training_meta.json").read_text())
        stage = (version_dir / "stage.txt").read_text().strip()
        metrics_locked = (version_dir / "metrics_locked.txt").exists()
        mv = ModelVersion(
            model_name=model_name,
            version=version,
            stage=stage,
            metrics=metrics,
            meta=meta,
            registered_at=datetime.strptime(version, "%Y%m%d-%H%M%S%f"),
            metrics_locked=metrics_locked,
        )
        self._versions.setdefault(model_name, {})[version] = mv
