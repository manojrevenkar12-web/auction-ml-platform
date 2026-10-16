"""On-demand feature transforms.

These run on raw request fields. The platform calls the same function when it
builds a training set and when the serving container answers a request, which is
what makes training/serving skew on these features impossible by construction.
Each function returns the declared features plus the lookup keys for batch groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return ((e.dt.year - s.dt.year) * 12 + (e.dt.month - s.dt.month)).astype("float64")


def _optional(raw: pd.DataFrame, name: str) -> pd.Series:
    if name in raw:
        return pd.to_numeric(raw[name], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def listing_features(raw: pd.DataFrame) -> pd.DataFrame:
    age = _months_between(raw["first_registration"], raw["as_of"]).clip(lower=0)
    make_model = raw["make"].astype(str) + " " + raw["model"].astype(str)
    fuel = raw["fuel"].astype(str)
    segment = raw["segment"].astype(str)
    mileage = raw["mileage_km"].astype("float64")
    return pd.DataFrame(
        {
            "make_model": make_model,
            "segment": segment,
            "fuel": fuel,
            "seller_country": raw["seller_country"].astype(str),
            "age_months": age,
            "mileage_km": mileage,
            "km_per_month": mileage / age.clip(lower=1),
            "damage_score": raw["damage_score"].astype("float64"),
            "power_kw": raw["power_kw"].astype("float64"),
            "start_price": _optional(raw, "start_price"),
            "relist_no": _optional(raw, "relist_no"),
            # lookup keys for batch feature groups
            "seller_id": raw["seller_id"].astype(str),
            "mm_fuel_key": make_model + "|" + fuel,
            "sf_key": segment + "|" + fuel,
            "auction_id": raw["auction_id"].astype(str) if "auction_id" in raw else pd.Series(np.nan, index=raw.index),
        },
        index=raw.index,
    )


def shipment_features(raw: pd.DataFrame) -> pd.DataFrame:
    lane = raw["origin_country"].astype(str) + ">" + raw["dest_country"].astype(str)
    return pd.DataFrame(
        {
            "lane": lane,
            "segment": raw["segment"].astype(str),
            "non_runner": (raw["damage_score"].astype("float64") >= 4).astype("float64"),
            "quote_month": pd.to_datetime(raw["as_of"]).dt.month.astype("float64"),
            "lane_key": lane,
        },
        index=raw.index,
    )
