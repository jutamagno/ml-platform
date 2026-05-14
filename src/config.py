from dataclasses import dataclass, field


@dataclass
class IngestorConfig:
    kafka_bootstrap: str = "localhost:9092"
    topic: str = "ad-events"
    dlq_topic: str = "ad-events-dlq"
    group_id: str = "ml-platform-ingestor"


@dataclass
class FeatureStoreConfig:
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    parquet_root: str = "/tmp/ml-platform/features"
    window_1h: int = 3_600
    window_24h: int = 86_400
    window_7d: int = 604_800


@dataclass
class SkewDetectorConfig:
    psi_warn_threshold: float = 0.2
    psi_alert_threshold: float = 0.25
    n_bins: int = 10


@dataclass
class TriggerConfig:
    event_count_threshold: int = 10_000
    schedule_cron: str = "0 2 * * *"  # 02:00 UTC daily


@dataclass
class TrainingConfig:
    registry_root: str = "/tmp/ml-platform/registry"
    validation_fraction: float = 0.2
    random_seed: int = 42
    conversion_window_s: int = 86_400
    min_training_examples: int = 100
    min_positive_examples: int = 10


@dataclass
class DeploymentConfig:
    min_shadow_hours: float = 2.0
    min_canary_hours: float = 24.0
    min_full_retention_hours: float = 48.0
    canary_fraction: float = 0.10
    metric_tolerance: float = 0.005  # new model may be this much worse


@dataclass
class RollbackConfig:
    check_interval_s: int = 300  # 5 minutes
    rollback_auc_drop: float = 0.01
    max_error_rate: float = 0.01
    max_latency_ms: float = 100.0


@dataclass
class PlatformConfig:
    ingestor: IngestorConfig = field(default_factory=IngestorConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    skew_detector: SkewDetectorConfig = field(default_factory=SkewDetectorConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    rollback: RollbackConfig = field(default_factory=RollbackConfig)


DEFAULT_CONFIG = PlatformConfig()
