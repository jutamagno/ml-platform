"""
Bridge: converts recsys-adtech auction events → ml-platform AdEvent dicts.

recsys-adtech produces (User, Item, AuctionResult) triples from simulated auctions.
ml-platform ingests flat dicts validated against AdEvent (Pydantic schema).

One auction → up to three events: impression + optional click + optional conversion.
Conversion events are written with a future timestamp (now + delay_hours) to
simulate the real-world delayed-reward window.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

# Resolve recsys-adtech imports
_RECSYS_ROOT = Path(__file__).parents[3] / "recsys-adtech"
for _p in (_RECSYS_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if TYPE_CHECKING:
    from shared.data.schemas import AuctionResult, Item, User


def auction_to_events(
    user: "User",
    item: "Item",
    result: "AuctionResult",
    occurred_at: datetime,
) -> list[dict]:
    """
    Converts one AuctionResult into a list of AdEvent-compatible dicts.

    impression → always emitted if result.impression is True
    click       → only if result.click
    conversion  → only if result.conversion, timestamped at occurred_at + delay
    """
    events: list[dict] = []

    if not result.impression:
        return events

    shared = {
        "user_id": user.user_id,
        "item_id": item.item_id,
        "session_id": None,
        "publisher_id": None,
        "campaign_id": item.advertiser_id,
    }

    events.append({
        **shared,
        "event_id": str(uuid.uuid4()),
        "event_type": "impression",
        "timestamp": occurred_at.isoformat(),
        "confirmed": True,
    })

    if result.click:
        events.append({
            **shared,
            "event_id": str(uuid.uuid4()),
            "event_type": "click",
            "timestamp": occurred_at.isoformat(),
            "confirmed": True,
        })

    if result.conversion:
        conversion_ts = occurred_at + timedelta(hours=result.conversion_delay_hours)
        events.append({
            **shared,
            "event_id": str(uuid.uuid4()),
            "event_type": "conversion",
            "timestamp": conversion_ts.isoformat(),
            "revenue": result.clearing_price,
            # Not confirmed until the conversion window closes
            "confirmed": result.conversion_delay_hours < 1.0,
        })

    return events


def build_feature_row(user: "User", item: "Item", hour: int) -> dict:
    """
    Calls recsys-adtech's build_feature_vector and returns a named dict
    compatible with ml-platform's FEATURE_COLS schema.

    The 114-dim recsys vector is richer than ml-platform's 6 aggregate
    features, so we expose it under column names feat_0..feat_113 and let
    the TrainingJob pick whatever columns it was configured with.
    """
    if str(_RECSYS_ROOT / "05-adtech" / "src") not in sys.path:
        sys.path.insert(0, str(_RECSYS_ROOT / "05-adtech" / "src"))

    from features import build_feature_vector  # resolved via sys.path
    vec = build_feature_vector(user, item, hour=hour)
    return {f"feat_{i}": float(v) for i, v in enumerate(vec)}
