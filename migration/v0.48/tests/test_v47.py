import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from inventory_v47 import _simulate_one, policy_grid, summarize, COST_SCENARIOS, LEAD_TIMES


def frame(demands, targets=None, weeks=None):
    if targets is None:
        targets = [10] * len(demands)
    if weeks is None:
        weeks = list(range(len(demands)))
    return pd.DataFrame({
        "series_id": ["sku"] * len(demands),
        "week": weeks,
        "date": [f"2011-01-{i+1:02d}" for i in range(len(demands))],
        "split": ["selection"] * len(demands),
        "demand": demands,
        "target": targets,
        "scale": [10.0] * len(demands),
    })


def test_lead_one_arrives_next_week_not_same_week():
    d = frame([10, 10, 10, 10])
    out = _simulate_one(d, 1, {"method": "target", "policy": "base_stock", "trigger_ratio": None}, COST_SCENARIOS["balanced"])
    assert out.order_qty.tolist() == [0, 10, 0, 10]
    assert out.arrival_qty.tolist() == [0, 0, 10, 0]
    assert out.lost_sales.tolist() == [0, 10, 0, 10]


def test_lead_two_arrives_after_two_periods():
    d = frame([10, 10, 10, 10, 10])
    out = _simulate_one(d, 2, {"method": "target", "policy": "base_stock", "trigger_ratio": None}, COST_SCENARIOS["balanced"])
    assert out.arrival_qty.tolist() == [0, 0, 0, 0, 10]
    assert out.lost_sales.tolist() == [0, 0, 10, 10, 0]


def test_ss_policy_waits_until_reorder_threshold():
    d = frame([2, 2, 2, 2], targets=[10, 10, 10, 10])
    out = _simulate_one(d, 1, {"method": "target", "policy": "sS", "trigger_ratio": 0.5}, COST_SCENARIOS["balanced"])
    # Initial stock 10; after two periods on-hand falls to 6, then order is triggered only at <=5.
    assert out.order_qty.iloc[1] == 0
    assert out.order_qty.iloc[2] == 0
    assert out.order_qty.iloc[3] == 6


def test_fixed_order_cost_only_when_order_is_placed():
    d = frame([10, 10, 10], targets=[10, 10, 10])
    costs = {"shortage_cost": 3.0, "holding_cost": 1.0, "fixed_order_cost": 7.0}
    out = _simulate_one(d, 1, {"method": "target", "policy": "base_stock", "trigger_ratio": None}, costs)
    assert out.fixed_order_cost.tolist() == [0.0, 7.0, 0.0]
    np.testing.assert_allclose(out.total_cost, out.holding_cost + out.lost_sales_cost + out.fixed_order_cost)


def test_inventory_balance_and_nonnegative_state():
    d = frame([3, 20, 0, 4, 18], targets=[10, 12, 15, 10, 14])
    out = _simulate_one(d, 2, {"method": "target", "policy": "base_stock", "trigger_ratio": None}, COST_SCENARIOS["balanced"])
    assert (out.on_hand_end >= 0).all()
    assert (out.order_qty >= 0).all()
    assert (out.lost_sales >= 0).all()
    assert (out.fulfilled + out.lost_sales == out.demand).all()
    assert (out.on_hand_start - out.fulfilled == out.on_hand_end).all()
    assert (out.loc[out.order_qty > 0, "arrival_week"] == out.loc[out.order_qty > 0, "week"] + 2).all()


def test_summarize_fill_rate_and_cost():
    d = frame([10, 10, 10], targets=[10, 10, 10])
    out = _simulate_one(d, 1, {"method": "target", "policy": "base_stock", "trigger_ratio": None}, COST_SCENARIOS["balanced"])
    s = summarize(out, "selection")
    assert 0 <= s["fill_rate"] <= 1
    assert s["n"] == 3
    assert s["mean_cost_per_sku_week"] == pytest.approx(out.total_cost.mean())


def test_policy_grid_contains_baselines_and_group_candidates():
    cfgs = policy_grid(2)
    assert len(cfgs) == 40
    assert any(x["method"] == "group_horizon_q75_L2" for x in cfgs)
    assert any(x["method"] == "global_horizon_q75_L2" for x in cfgs)
    assert {x["policy"] for x in cfgs} == {"base_stock", "sS"}


def test_invalid_policy_rejected():
    d = frame([1, 2, 3])
    with pytest.raises(ValueError):
        _simulate_one(d, 1, {"method": "target", "policy": "unknown", "trigger_ratio": None}, COST_SCENARIOS["balanced"])


def test_cost_scenarios_are_positive_and_distinct():
    assert set(COST_SCENARIOS) == {"balanced", "service_priority", "capital_pressure", "order_setup_cost"}
    for c in COST_SCENARIOS.values():
        assert c["shortage_cost"] > 0 and c["holding_cost"] > 0 and c["fixed_order_cost"] >= 0
    assert set(LEAD_TIMES) == {1, 2, 4}


def test_missing_forecast_column_rejected():
    d = frame([1, 2, 3])
    with pytest.raises(ValueError):
        _simulate_one(d, 1, {"method": "missing", "policy": "base_stock", "trigger_ratio": None}, COST_SCENARIOS["balanced"])


def test_prepared_panel_hash_and_provenance(tmp_path, monkeypatch):
    import json, hashlib
    import inventory_v47 as module
    panel_path = tmp_path / "panel.csv"
    quality_path = tmp_path / "quality.json"
    raw = pd.DataFrame({
        "series_id": ["a", "a", "b"],
        "time_index": [0, 1, 0],
        "date": ["2011-01-02", "2011-01-09", "2011-01-02"],
        "demand": [3.0, 4.0, 8.0],
    })
    raw.to_csv(panel_path, index=False)
    quality = {
        "zip_sha256": "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb",
        "xlsx_sha256": "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980",
        "selected_panel_rows": 3,
        "selected_series": 2,
    }
    quality_path.write_text(json.dumps(quality))
    monkeypatch.setattr(module, "PREPARED_PANEL_SHA", hashlib.sha256(panel_path.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "PREPARED_QUALITY_SHA", hashlib.sha256(quality_path.read_bytes()).hexdigest())
    panel, meta = module.load_prepared_panel(panel_path, quality_path)
    assert len(panel) == 3 and panel.series_id.nunique() == 2
    assert meta["prepared_panel_sha256"] == module.PREPARED_PANEL_SHA


def test_prepared_panel_rejects_modified_bytes(tmp_path):
    from inventory_v47 import load_prepared_panel
    bad = tmp_path / "panel.csv"
    quality = tmp_path / "quality.json"
    bad.write_text("series_id,date,demand\na,2011-01-02,1\n")
    quality.write_text("{}")
    with pytest.raises(ValueError):
        load_prepared_panel(bad, quality)
