"""Generate a synthetic European B2B used-car auction market into a DuckDB warehouse.

The generator plants the things a production ML platform has to cope with:

* relisted vehicles (the same VIN appears in several auctions),
* labels that arrive late (hammer price at auction end, transport cost on delivery),
* a market shock (EV wholesale prices fall ~20% over three weeks from 2 Mar 2026),
* a new seller country onboarded mid-stream (PL from 13 Apr 2026, higher mileage),
* a post-event field (``n_bids``) that looks like a great feature and is not.

Everything is driven by one seed so the warehouse is bit-for-bit reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

START = date(2025, 1, 1)
END = date(2026, 7, 1)  # exclusive
EV_SHOCK_START = date(2026, 3, 2)
EV_SHOCK_DAYS = 21
EV_SHOCK_DEPTH = 0.20
PL_ONBOARD = date(2026, 4, 13)

# make, model, segment, new price (EUR), fuels, power kW
CATALOG: list[tuple[str, str, str, int, tuple[str, ...], int]] = [
    ("VW", "Golf", "compact", 31000, ("petrol", "diesel", "hybrid"), 96),
    ("VW", "Polo", "small", 22000, ("petrol",), 70),
    ("VW", "ID.3", "compact", 39000, ("ev",), 150),
    ("BMW", "3 Series", "premium", 49000, ("petrol", "diesel", "hybrid"), 135),
    ("Mercedes", "C-Class", "premium", 51000, ("petrol", "diesel", "hybrid"), 150),
    ("Tesla", "Model 3", "premium", 46000, ("ev",), 208),
    ("Audi", "Q5", "suv", 56000, ("petrol", "diesel", "hybrid"), 150),
    ("Skoda", "Octavia", "compact", 29000, ("petrol", "diesel"), 110),
    ("Renault", "Clio", "small", 19500, ("petrol", "diesel"), 67),
    ("Hyundai", "Kona", "suv", 33000, ("petrol", "hybrid", "ev"), 100),
    ("Ford", "Transit", "van", 41000, ("diesel",), 96),
    ("Opel", "Corsa", "small", 20500, ("petrol", "ev"), 74),
]
SEGMENT_DEMAND = {"small": 1.10, "compact": 1.05, "premium": 0.95, "suv": 1.0, "van": 0.85}
SEGMENT_SIZE = {"small": 0.90, "compact": 1.0, "premium": 1.05, "suv": 1.15, "van": 1.40}
COUNTRY_PRICE = {"DE": 1.00, "AT": 1.02, "NL": 0.98, "FR": 0.95, "IT": 0.93, "PL": 0.86}
SELLER_COUNTRIES = ["DE"] * 12 + ["FR"] * 3 + ["IT"] * 2 + ["NL"] * 2 + ["AT"] * 1
BUYER_COUNTRIES = ["DE", "FR", "IT", "NL", "AT", "PL", "ES", "BE", "CZ", "DK"]
BUYER_WEIGHTS = [0.38, 0.10, 0.09, 0.08, 0.06, 0.10, 0.06, 0.05, 0.05, 0.03]
CENTROIDS = {  # rough lat/lon of national road-freight hubs
    "DE": (51.0, 10.0),
    "FR": (47.0, 2.5),
    "IT": (44.5, 10.5),
    "NL": (52.2, 5.3),
    "AT": (47.6, 14.1),
    "PL": (52.0, 19.0),
    "ES": (40.4, -3.7),
    "BE": (50.6, 4.5),
    "CZ": (49.8, 15.5),
    "DK": (56.0, 9.5),
}


def road_km(a: str, b: str) -> float:
    """Great-circle distance between hubs times a road-detour factor."""
    if a == b:
        return 250.0
    (la1, lo1), (la2, lo2) = CENTROIDS[a], CENTROIDS[b]
    p1, p2 = np.radians(la1), np.radians(la2)
    dl = np.radians(lo2 - lo1)
    dp = p2 - p1
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(h)) * 1.25)


def fuel_index(fuel: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Wholesale price level per fuel type over time (1.0 = Jan 2025)."""
    months = (pd.to_datetime(day) - pd.Timestamp(START)).days.to_numpy() / 30.4
    drift = {"petrol": -0.001, "diesel": -0.004, "hybrid": 0.001, "ev": -0.006}
    idx = np.ones(len(fuel))
    for f, d in drift.items():
        m = fuel == f
        idx[m] = (1 + d) ** months[m]
    shock_days = (pd.to_datetime(day) - pd.Timestamp(EV_SHOCK_START)).days.to_numpy()
    progress = np.clip(shock_days / EV_SHOCK_DAYS, 0, 1)
    ev = fuel == "ev"
    idx[ev] *= 1 - EV_SHOCK_DEPTH * progress[ev]
    return idx


@dataclass(frozen=True)
class MarketSize:
    listings_per_weekday: float = 115.0
    sellers: int = 450
    pl_sellers: int = 45
    pl_listings_per_weekday: float = 18.0


def _sellers(rng: np.random.Generator, size: MarketSize) -> pd.DataFrame:
    n = size.sellers + size.pl_sellers
    countries = list(rng.choice(SELLER_COUNTRIES, size.sellers)) + ["PL"] * size.pl_sellers
    kinds = rng.choice(["dealer", "leasing", "oem"], n, p=[0.70, 0.22, 0.08])
    return pd.DataFrame(
        {
            "seller_id": [f"S{i:04d}" for i in range(n)],
            "country": countries,
            "seller_type": kinds,
            "onboarded_at": [START] * size.sellers + [PL_ONBOARD] * size.pl_sellers,
            # latent traits, never exported to the warehouse
            "_ambition": rng.normal(0.0, 0.07, n),
            "_appeal": rng.lognormal(0.0, 0.25, n),
        }
    )


def _first_listings(rng: np.random.Generator, sellers: pd.DataFrame, size: MarketSize) -> pd.DataFrame:
    days = pd.date_range(START, END - timedelta(days=1), freq="D")
    rows = []
    legacy = sellers[sellers.country != "PL"]
    pl = sellers[sellers.country == "PL"]
    for d in days:
        weekday = d.weekday() < 5
        n = rng.poisson(size.listings_per_weekday if weekday else size.listings_per_weekday * 0.25)
        picks = legacy.sample(n, replace=True, random_state=int(rng.integers(1 << 31)))
        if d.date() >= PL_ONBOARD:
            m = rng.poisson(size.pl_listings_per_weekday if weekday else size.pl_listings_per_weekday * 0.25)
            picks = pd.concat([picks, pl.sample(m, replace=True, random_state=int(rng.integers(1 << 31)))])
        hours = rng.uniform(7, 19, len(picks))
        rows.append(pd.DataFrame({"seller_id": picks.seller_id.to_numpy(), "listed_at": d + pd.to_timedelta(hours, unit="h")}))
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(sellers[["seller_id", "country", "_ambition", "_appeal"]], on="seller_id", how="left")
    n = len(out)
    cat_idx = rng.integers(0, len(CATALOG), n)
    out["make"] = [CATALOG[i][0] for i in cat_idx]
    out["model"] = [CATALOG[i][1] for i in cat_idx]
    out["segment"] = [CATALOG[i][2] for i in cat_idx]
    out["_new_price"] = [CATALOG[i][3] for i in cat_idx]
    out["fuel"] = [str(rng.choice(CATALOG[i][4])) for i in cat_idx]
    out["power_kw"] = [CATALOG[i][5] for i in cat_idx] * rng.choice([0.85, 1.0, 1.2], n, p=[0.25, 0.5, 0.25])
    age_months = rng.integers(10, 145, n)
    out["first_registration"] = (out.listed_at.dt.to_period("M") - age_months).dt.to_timestamp().dt.date
    km_rate = rng.lognormal(np.log(1350), 0.35, n) * np.where(out.country == "PL", 1.4, 1.0)
    out["mileage_km"] = np.round(age_months * km_rate, -2)
    out["damage_score"] = np.minimum(rng.poisson(0.9, n), 5)
    out["vin"] = [f"WV{rng.integers(10**12, 10**13)}" for _ in range(n)]
    out["relist_no"] = 0
    return out


def _true_value(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    age = (
        (df.listed_at.dt.year - pd.to_datetime(df.first_registration).dt.year) * 12
        + (df.listed_at.dt.month - pd.to_datetime(df.first_registration).dt.month)
    ).to_numpy()
    v = df._new_price.to_numpy() * 0.82 * np.exp(-0.0115 * age)
    v *= np.exp(-0.0000032 * df.mileage_km.to_numpy())
    v *= 1 - 0.075 * df.damage_score.to_numpy()
    v *= (df.power_kw.to_numpy() / df.groupby(["make", "model"]).power_kw.transform("median").to_numpy()) ** 0.25
    v *= df.country.map(COUNTRY_PRICE).to_numpy()
    v *= fuel_index(df.fuel.to_numpy(), df.listed_at.to_numpy())
    season = 1 + 0.02 * np.sin(2 * np.pi * (df.listed_at.dt.dayofyear.to_numpy() - 80) / 365)
    v *= season
    return v * rng.lognormal(0.0, 0.07, len(df))


def _run_auctions(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    value = df["_value"].to_numpy()
    ask = df["start_price"].to_numpy()
    ratio = ask / value
    lam = 4.0 * np.exp(-5.0 * (ratio - 0.90)) * df.segment.map(SEGMENT_DEMAND).to_numpy() * df._appeal.to_numpy()
    lam *= np.where(df.fuel.to_numpy() == "ev", 0.85, 1.0)
    bidders = rng.poisson(np.clip(lam, 0.05, 40))
    hammer = np.full(len(df), np.nan)
    sold = np.zeros(len(df), dtype=bool)
    n_bids = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        k = bidders[i]
        if k == 0:
            continue
        vals = np.sort(value[i] * rng.lognormal(0.0, 0.06, k))[::-1]
        if vals[0] < ask[i]:
            n_bids[i] = int(rng.integers(0, k + 1))
            continue
        second = vals[1] if k > 1 else ask[i]
        hammer[i] = max(ask[i], second + 100)
        sold[i] = True
        n_bids[i] = int(k + rng.poisson(1.5 * k))
    df["sold"] = sold
    df["hammer_price"] = np.round(hammer, -1)
    df["n_bids"] = n_bids
    return df


def generate(seed: int = 7, size: MarketSize | None = None) -> dict[str, pd.DataFrame]:
    """Return the warehouse tables as DataFrames (latent columns stripped)."""
    size = size or MarketSize()
    rng = np.random.default_rng(seed)
    sellers = _sellers(rng, size)
    first = _first_listings(rng, sellers, size)
    first["_value"] = _true_value(first, rng)
    first["start_price"] = np.round(first._value * np.clip(0.97 + first._ambition + rng.normal(0, 0.06, len(first)), 0.70, 1.25), -2)
    first["listed_at"] = first.listed_at
    first["ended_at"] = first.listed_at + pd.to_timedelta(rng.choice([24, 48, 72], len(first), p=[0.2, 0.6, 0.2]), unit="h")

    rounds = [_run_auctions(first, rng)]
    current = rounds[-1]
    for relist in (1, 2):
        unsold = current[~current.sold]
        unsold = unsold[rng.random(len(unsold)) < 0.7].copy()
        gap = pd.to_timedelta(rng.uniform(3, 10, len(unsold)), unit="D")
        unsold["listed_at"] = unsold.ended_at + gap
        unsold["ended_at"] = unsold.listed_at + pd.Timedelta(hours=48)
        unsold["start_price"] = np.round(unsold.start_price * rng.uniform(0.90, 0.96, len(unsold)), -2)
        unsold["relist_no"] = relist
        unsold = unsold[unsold.listed_at < pd.Timestamp(END)]
        current = _run_auctions(unsold.drop(columns=["sold", "hammer_price", "n_bids"]), rng)
        rounds.append(current)
    auctions = pd.concat(rounds, ignore_index=True).sort_values("listed_at", kind="stable").reset_index(drop=True)
    auctions["auction_id"] = [f"A{i:06d}" for i in range(len(auctions))]
    # auctions still running at the warehouse snapshot have no outcome yet
    snapshot = pd.Timestamp(END)
    running = auctions.ended_at >= snapshot
    auctions.loc[running, ["sold", "hammer_price"]] = [False, np.nan]
    auctions.loc[running, "n_bids"] = 0
    auctions.loc[running, "ended_at"] = pd.NaT

    vehicles = (
        auctions.sort_values("relist_no")
        .drop_duplicates("vin")[["vin", "make", "model", "segment", "fuel", "first_registration", "power_kw"]]
        .reset_index(drop=True)
    )
    vehicles["power_kw"] = vehicles.power_kw.round().astype(int)

    sold = auctions[auctions.sold].copy()
    dest = rng.choice(BUYER_COUNTRIES, len(sold), p=BUYER_WEIGHTS)
    km = np.array([road_km(o, d) for o, d in zip(sold.country, dest, strict=True)])
    months = (sold.ended_at - pd.Timestamp(START)).dt.days.to_numpy() / 30.4
    surcharge = 1.01**months
    cost = (95 + 0.55 * km) * sold.segment.map(SEGMENT_SIZE).to_numpy()
    cost *= np.where(sold.damage_score.to_numpy() >= 4, 1.35, 1.0) * surcharge
    cost *= rng.lognormal(0.0, 0.10, len(sold))
    delivered = sold.ended_at + pd.to_timedelta(rng.uniform(2.5, 4.0, len(sold)) + km / 250, unit="D")
    transports = pd.DataFrame(
        {
            "auction_id": sold.auction_id.to_numpy(),
            "origin_country": sold.country.to_numpy(),
            "dest_country": dest,
            "booked_at": sold.ended_at.to_numpy(),
            "delivered_at": delivered.to_numpy(),
            "cost_eur": np.round(cost, 2),
        }
    )
    in_flight = transports.delivered_at >= snapshot
    transports.loc[in_flight, ["delivered_at", "cost_eur"]] = [pd.NaT, np.nan]

    auctions_out = auctions[
        [
            "auction_id",
            "vin",
            "seller_id",
            "listed_at",
            "ended_at",
            "mileage_km",
            "damage_score",
            "start_price",
            "relist_no",
            "n_bids",
            "sold",
            "hammer_price",
        ]
    ].copy()
    auctions_out["mileage_km"] = auctions_out.mileage_km.astype(int)
    auctions_out["damage_score"] = auctions_out.damage_score.astype(int)
    sellers_out = sellers[["seller_id", "country", "seller_type", "onboarded_at"]].copy()
    origins = sorted(set(SELLER_COUNTRIES) | {"PL"})
    lanes = pd.DataFrame(
        [(f"{o}>{d}", o, d, round(road_km(o, d), 1)) for o in origins for d in BUYER_COUNTRIES],
        columns=["lane_key", "origin_country", "dest_country", "distance_km"],
    )
    return {"sellers": sellers_out, "vehicles": vehicles, "auctions": auctions_out, "transports": transports, "lanes": lanes}


def write_warehouse(path: Path, seed: int = 7, size: MarketSize | None = None) -> dict[str, int]:
    """Write the synthetic market into a DuckDB file; returns row counts."""
    tables = generate(seed, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    try:
        for name, df in tables.items():
            con.register("_df", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
            con.unregister("_df")
        con.execute(
            "CREATE TABLE _meta AS SELECT ? AS seed, ?::TIMESTAMP AS snapshot_at",
            [seed, datetime.combine(END, datetime.min.time())],
        )
    finally:
        con.close()
    return {k: len(v) for k, v in tables.items()}
