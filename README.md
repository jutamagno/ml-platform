# ml-platform

A production ML lifecycle system for recommendation/ranking models.
Answers the question: **"How do you keep a model working correctly in production, weeks after the first deploy?"**

---

## The Problem

A recommendation model trained once degrades over time because:

1. **Data drift** — user behavior shifts; feature distributions at serving time diverge from training
2. **Label drift** — CTR/CVR change due to seasonality, new advertisers, and creative fatigue
3. **Feedback loops** — the model influences what users see, which influences future training data, amplifying biases

Without a continuous training and deployment system, these problems accumulate silently until a business metric drops enough to trigger a manual investigation. By then, significant revenue has been lost.

---

## System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              ml-platform                 │
                    │                                          │
  raw events        │  ┌──────────┐    ┌───────────────────┐  │
  (Kafka topic) ────┼─▶│ Ingestor │───▶│  Feature Store    │  │
                    │  └──────────┘    │  (Redis + Parquet) │  │
                    │                  └────────┬──────────┘  │
                    │                           │              │
                    │                  ┌────────▼──────────┐  │
                    │                  │  Training Trigger  │  │
                    │                  │  (event count or   │  │
                    │                  │   schedule-based)  │  │
                    │                  └────────┬──────────┘  │
                    │                           │              │
                    │                  ┌────────▼──────────┐  │
                    │                  │  Training Job      │  │
                    │                  │  (LightGBM /       │  │
                    │                  │   PyTorch)         │  │
                    │                  └────────┬──────────┘  │
                    │                           │              │
                    │                  ┌────────▼──────────┐  │
                    │                  │  Model Registry    │  │
                    │                  │  (versioned        │  │
                    │                  │   artifacts +      │  │
                    │                  │   metrics history) │  │
                    │                  └────────┬──────────┘  │
                    │                           │              │
                    │                  ┌────────▼──────────┐  │
                    │                  │  Deployment Engine │  │
                    │                  │  shadow → canary   │  │
                    │                  │  → full → rollback │  │
                    │                  └────────┬──────────┘  │
                    │                           │              │
                    │                  ┌────────▼──────────┐  │
  serving ◀─────────┼──────────────────│  Serving Proxy     │  │
  requests          │                  │  (routes traffic   │  │
                    │                  │   per stage)       │  │
                    │                  └───────────────────┘  │
                    └─────────────────────────────────────────┘
```

---

## Deployment Stages

```
SHADOW ──────────────────────────────────────────────────────────────────────────┐
  New model runs in parallel.                                                    │
  Results discarded. Crashes caught before real users are affected.             │
  Minimum duration: 2h. Advance condition: shadow AUC ≥ production - tolerance. │
         │                                                                       │
         ▼                                                                       │
CANARY ──────────────────────────────────────────────────────────────────────────┤
  10% of traffic routed to new model; 90% to production.                        │ ROLLBACK
  Routing is deterministic by user_id hash.                                     │ if metrics
  Minimum duration: 24h. Advance condition: online metrics ≥ production.        │ degrade
         │                                                                       │
         ▼                                                                       │
FULL ────────────────────────────────────────────────────────────────────────────┤
  100% traffic to new model.                                                     │
  Old model retained 48h then archived.                                         │
         │                                                                       │
         └──────────────────────────────── ROLLED_BACK ◀──────────────────────────┘
```

---

## Design Decisions

**Why SHADOW before CANARY?**
Shadow mode runs the new model against all production traffic without returning its results to users. This catches crashes, serialization errors, and extreme latency regressions before any real user is affected. CANARY without a prior SHADOW stage would expose bugs directly to 10% of users.

**Why deterministic user routing in CANARY by user_id hash?**
A user who gets recommendation A on one request and recommendation B on the next (because the traffic split is random per request) has an inconsistent experience and may attribute the inconsistency to the product. Hashing on user_id guarantees the same user always sees the same model during the canary period.

**Why three independent rollback conditions?**
- **AUC drop**: catches statistical degradation that's invisible in infrastructure metrics — a model can be wrong in a subtle way without throwing errors.
- **Error rate**: catches model crashes, serialization failures, and timeouts that AUC can't detect (no predictions → no AUC).
- **p99 latency**: catches performance regressions that increase serving cost and timeout risk before they affect enough users to degrade AUC.

**Why PSI over KL divergence for skew detection?**
KL divergence is undefined when the reference distribution has zero-count bins, which happens often in production with sparse categorical features. PSI is symmetric and uses additive smoothing, giving a stable score even with sparse bins. PSI > 0.2 is an established industry threshold for meaningful drift.

**Why stateless models behind a routing proxy?**
If models knew their own deployment stage, every model would need rollout logic, shadow logging, and routing state — duplicating non-ML complexity across every model. By placing all routing in the proxy, models remain simple HTTP endpoints that just do inference. Routing logic is isolated and independently testable.

---

## Connection to `recsys-adtech`

`recsys-adtech` builds the recommendation models. `ml-platform` is the system that would operate those models in production — handling retraining when their feature distributions drift, staged deployment to safely replace one version with another, and automated rollback when a new version degrades.

---

## Quickstart

```bash
# Start Kafka + Redis
docker-compose up -d

# Install dependencies
pip install -e ".[dev]"

# Run the full lifecycle simulation
python scripts/run_full_cycle.py

# Run tests
pytest tests/ -v
```

Expected output from `run_full_cycle.py`:

```
[INGESTOR  ]  20000 events ingested; 0 dead-letter
[TRIGGER   ]  Fired: event_count=5000
[TRAINING  ]  v1 trained: AUC=0.734, ECE=0.021, n=4821 confirmed examples
[DEPLOY    ]  v1: SHADOW
[DEPLOY    ]  v1: CANARY(10%)
[DEPLOY    ]  v1 → FULL
[SKEW      ]  feature 'user_clicks_24h': PSI=0.31 (WARNING)
[TRIGGER   ]  Fired: event_count=5000
[TRAINING  ]  v2 trained: AUC=0.741, ECE=0.019, n=5103 confirmed examples
[DEPLOY    ]  v2: SHADOW
[DEPLOY    ]  v2: CANARY(10%)
[ROLLBACK  ]  Triggered: p99_latency=143ms > 100ms threshold
[DEPLOY    ]  Rolled back to v1
[REGISTRY  ]  versions: v1(production), v2(rolled_back)
```

---

## File Structure

```
ml-platform/
├── src/
│   ├── ingestor/        # EventIngestor, DeadLetterHandler, Pydantic schemas
│   ├── feature_store/   # FeatureStore (Redis + Parquet), SkewDetector (PSI)
│   ├── training/        # TrainingTrigger, TrainingJob, ModelAdapter protocol
│   ├── registry/        # ModelRegistry — versioned artifacts + metrics
│   ├── deployment/      # DeploymentEngine, RollbackMonitor, ServingProxy
│   └── config.py        # All thresholds and constants in one place
├── tests/               # One test file per component
├── scripts/
│   └── run_full_cycle.py
└── docker-compose.yml   # Kafka + Redis
```
