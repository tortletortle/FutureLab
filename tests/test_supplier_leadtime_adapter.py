import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from supplier_leadtime_adapter import normalize_supplier_orders, estimate_lead_time_distribution


def sample_orders():
    return pd.DataFrame({
        "order_id": range(1, 41),
        "supplier_id": ["A"] * 20 + ["B"] * 20,
        "order_date": ["2025-01-01"] * 40,
        "receipt_date": (
            ["2025-01-05"] * 10 + ["2025-01-12"] * 5 + ["2025-01-20"] * 3 + ["2025-02-01"] * 2
            + ["2025-01-08"] * 10 + ["2025-01-15"] * 5 + ["2025-01-22"] * 3 + ["2025-02-05"] * 2
        ),
        "quantity": [10] * 40,
        "unit_cost": [2.0] * 40,
    })


def test_normalize_supplier_orders_computes_completed_and_pending():
    frame = sample_orders()
    frame.loc[0, "receipt_date"] = None
    out = normalize_supplier_orders(frame)
    assert out.completed.sum() == 39
    assert pd.isna(out.loc[0, "lead_days"])


def test_supplier_distribution_uses_only_orders_before_as_of():
    frame = sample_orders()
    frame.loc[len(frame)] = [41, "A", "2025-03-01", "2025-03-30", 10, 2.0]
    out = estimate_lead_time_distribution(frame, "2025-02-01", supplier_id="A")
    assert out["sample_orders"] == 20
    assert out["used_future_orders"] is False


def test_supplier_small_sample_shrinks_to_global():
    frame = sample_orders()
    out = estimate_lead_time_distribution(frame, "2025-02-01", supplier_id="A", min_orders=30)
    assert out["evidence_level"] == "supplier_shrunk_to_global"
    assert sum(out["probabilities"].values()) == pytest.approx(1.0)


def test_supplier_distribution_reports_four_plus_week_bucket():
    frame = sample_orders()
    frame.loc[0, "receipt_date"] = "2025-03-01"
    out = estimate_lead_time_distribution(frame, "2025-04-01", supplier_id="A")
    assert out["four_plus_week_count"] == 3
    assert out["over_28_day_count"] == 3
    assert out["compatible_with_v48_four_week_cap"] is False
    assert out["probabilities"][4] > 0


def test_duplicate_order_ids_rejected():
    frame = sample_orders()
    frame.loc[1, "order_id"] = frame.loc[0, "order_id"]
    with pytest.raises(ValueError):
        normalize_supplier_orders(frame)


def test_receipt_before_order_rejected():
    frame = sample_orders()
    frame.loc[0, "receipt_date"] = "2024-12-31"
    with pytest.raises(ValueError):
        normalize_supplier_orders(frame)


def test_missing_required_supplier_columns_rejected():
    with pytest.raises(ValueError):
        normalize_supplier_orders(pd.DataFrame({"supplier_id": ["A"], "order_date": ["2025-01-01"]}))
