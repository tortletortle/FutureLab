"""Import and estimate supplier lead-time distributions from actual PO exports.

This adapter is intentionally separate from the V0.48 benchmark: Online Retail II
contains no supplier purchase orders. No synthetic/demo dataset is treated as
empirical supplier evidence.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"supplier_id", "order_date", "receipt_date"}


def normalize_supplier_orders(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("supplier order table must be a non-empty DataFrame")
    d = frame.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    missing = REQUIRED_COLUMNS - set(d.columns)
    if missing:
        raise ValueError(f"missing required supplier-order columns: {sorted(missing)}")
    if d.supplier_id.isna().any():
        raise ValueError("supplier_id cannot be null")
    if "order_id" in d.columns and d.order_id.notna().any():
        ids = d.loc[d.order_id.notna(), "order_id"]
        if ids.duplicated().any():
            raise ValueError("duplicate order_id")
    d["order_date"] = pd.to_datetime(d.order_date, errors="coerce", utc=True)
    d["receipt_date"] = pd.to_datetime(d.receipt_date, errors="coerce", utc=True)
    if d.order_date.isna().any():
        raise ValueError("order_date contains invalid or missing values")
    completed = d.receipt_date.notna()
    if (d.loc[completed, "receipt_date"] < d.loc[completed, "order_date"]).any():
        raise ValueError("receipt_date cannot precede order_date")
    if "quantity" in d.columns:
        q = pd.to_numeric(d.quantity, errors="coerce")
        if q.notna().any() and ((q.dropna() <= 0).any() or q.isna().any()):
            raise ValueError("quantity must be positive when supplied")
        d["quantity"] = q
    if "unit_cost" in d.columns:
        c = pd.to_numeric(d.unit_cost, errors="coerce")
        if c.notna().any() and ((c.dropna() <= 0).any() or c.isna().any()):
            raise ValueError("unit_cost must be positive when supplied")
        d["unit_cost"] = c
    d["completed"] = completed
    d["lead_days"] = np.where(
        completed,
        (d.receipt_date - d.order_date).dt.total_seconds() / 86400.0,
        np.nan,
    )
    return d


def load_supplier_orders(path: str | Path) -> pd.DataFrame:
    return normalize_supplier_orders(pd.read_csv(path))


def estimate_lead_time_distribution(
    orders: pd.DataFrame,
    as_of: str | pd.Timestamp,
    supplier_id: str | int | None = None,
    min_orders: int = 30,
    prior_strength: float = 10.0,
) -> dict:
    """Estimate 1,2,3,4+ week buckets with supplier-to-global shrinkage.

    Only completed orders placed strictly before `as_of` are used. Bucket 4
    means four or more weeks; it must not be interpreted as exactly four weeks.
    """
    if min_orders < 1 or prior_strength < 0:
        raise ValueError("invalid minimum sample or prior strength")
    d = normalize_supplier_orders(orders)
    cutoff = pd.to_datetime(as_of, utc=True)
    hist = d[(d.order_date < cutoff) & d.completed].copy()
    if hist.empty:
        raise ValueError("no completed historical supplier orders before as_of")
    hist["lead_week_bucket"] = np.maximum(1, np.ceil(np.maximum(hist.lead_days, 1.0) / 7.0)).astype(int).clip(upper=4)
    global_counts = hist.lead_week_bucket.value_counts().reindex([1, 2, 3, 4], fill_value=0).to_numpy(float)
    global_prob = global_counts / global_counts.sum()

    if supplier_id is None:
        sample = hist
        evidence_level = "global_empirical"
    else:
        sample = hist[hist.supplier_id.astype(str) == str(supplier_id)]
        evidence_level = "supplier_empirical" if len(sample) >= min_orders else "supplier_shrunk_to_global"
    counts = sample.lead_week_bucket.value_counts().reindex([1, 2, 3, 4], fill_value=0).to_numpy(float)
    n = int(counts.sum())
    if supplier_id is not None and n == 0:
        evidence_level = "global_fallback"
    if supplier_id is None:
        probs = global_prob
    elif n == 0:
        probs = global_prob
    else:
        probs = (counts + prior_strength * global_prob) / (n + prior_strength)
    evidence_sample = sample if n else hist
    over_28_days = int((evidence_sample.lead_days > 28).sum())
    return {
        "supplier_id": None if supplier_id is None else str(supplier_id),
        "as_of": str(cutoff.date()),
        "probabilities": {i + 1: float(probs[i]) for i in range(4)},
        "sample_orders": n,
        "global_sample_orders": int(len(hist)),
        "evidence_level": evidence_level,
        "lead_days_mean": float(sample.lead_days.mean()) if n else float(hist.lead_days.mean()),
        "lead_days_p90": float(sample.lead_days.quantile(0.90)) if n else float(hist.lead_days.quantile(0.90)),
        "four_plus_week_share": float(probs[3]),
        "four_plus_week_count": int((evidence_sample.lead_week_bucket == 4).sum()),
        "over_28_day_count": over_28_days,
        "compatible_with_v48_four_week_cap": over_28_days == 0,
        "used_future_orders": False,
        "bucket_semantics": {"1": "1-7 days", "2": "8-14 days", "3": "15-21 days", "4": "22-28 days; values over 28 days are flagged as incompatible rather than silently truncated"},
    }
