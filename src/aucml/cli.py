"""Command line entry point. Every command takes a model name and works for any model."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from aucml import gate as gate_mod
from aucml import review as review_mod
from aucml.config import Catalog, ModelSpec, discover_specs
from aucml.featurestore import OfflineStore, OnlineStore, materialize_offline
from aucml.packaging import package, unpack
from aucml.pipeline import Workspace, run
from aucml.registry import PromotionError, Registry


def _ctx(args: argparse.Namespace) -> tuple[Workspace, Catalog]:
    root = Path(args.root)
    return Workspace(root), Catalog.load(root / "feature_repo")


def _spec(args: argparse.Namespace) -> ModelSpec:
    root = Path(args.root)
    for s in discover_specs(root / "models", include_experiments=True):
        if s.name == args.model:
            return s
    raise SystemExit(f"unknown model {args.model!r}")


def _latest_run(ws: Workspace, model: str) -> Path:
    runs = sorted((ws.runs / model).iterdir(), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"no runs for {model}")
    return runs[-1]


def cmd_materialize(args: argparse.Namespace) -> None:
    ws, cat = _ctx(args)
    con = duckdb.connect(str(ws.warehouse), read_only=True)
    try:
        snap = con.execute("SELECT snapshot_at FROM _meta").fetchone()
        first = con.execute("SELECT min(listed_at) FROM auctions").fetchone()
    finally:
        con.close()
    assert snap and first
    start = pd.Timestamp(first[0]).normalize().date()
    end = pd.Timestamp(snap[0]).date()
    print(json.dumps(materialize_offline(cat, ws.warehouse, ws.store, str(start), str(end)), indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    ws, cat = _ctx(args)
    r = run(_spec(args), cat, ws)
    print(
        json.dumps(
            {"run_id": r.run_id, "dir": str(r.dir), "metrics": r.evaluation["metrics"], "baseline": r.evaluation["baseline"], "timings_s": r.timings},
            indent=2,
        )
    )


def cmd_review(args: argparse.Namespace) -> None:
    ws, cat = _ctx(args)
    spec = _spec(args)
    rd = Path(args.run) if args.run else _latest_run(ws, spec.name)
    findings = review_mod.review(spec, cat, pd.read_parquet(rd / "train.parquet"), pd.read_parquet(rd / "test.parquet"))
    md = review_mod.to_markdown(spec, findings)
    (rd / "review.md").write_text(md)
    print(md)
    if review_mod.verdict(findings) == "block":
        sys.exit(2)


def release(spec: ModelSpec, cat: Catalog, ws: Workspace, lane: str = "prod", approved_by: str | None = None) -> dict[str, Any]:
    """run -> review -> package -> register -> gate vs current champion -> promote."""
    r = run(spec, cat, ws)
    findings = review_mod.review(spec, cat, r.train, r.test)
    reg = Registry(ws.registry)
    with tempfile.TemporaryDirectory() as tmp:
        tar, manifest = package(r.dir, spec, cat, Path(tmp))
        reg.register(spec.name, r.run_id, tar, manifest, review_mod.as_dict(findings))
    champion = reg.lanes(spec.name).get("prod")
    champ_pkg = reg.vdir(spec.name, champion) / "model.tar.gz" if champion and champion != r.run_id else None
    decision, _ = gate_mod.decide(spec, cat, ws, r.dir, reg.vdir(spec.name, r.run_id) / "model.tar.gz", champ_pkg)
    reg.record_gate(spec.name, r.run_id, decision, gate_mod.to_markdown(decision))
    promoted: list[str] = []
    try:
        reg.promote(spec.name, r.run_id, "dev", reason="release")
        promoted.append("dev")
        if decision["decision"] == "approve" and lane in ("staging", "prod"):
            reg.promote(spec.name, r.run_id, "staging", reason="gate approved")
            promoted.append("staging")
            if lane == "prod":
                reg.promote(spec.name, r.run_id, "prod", approved_by=approved_by, reason="gate approved")
                promoted.append("prod")
    except PromotionError as exc:
        promoted.append(f"stopped: {exc}")
    return {"version": r.run_id, "review": review_mod.verdict(findings), "decision": decision, "promoted": promoted, "champion": champion}


def cmd_release(args: argparse.Namespace) -> None:
    ws, cat = _ctx(args)
    out = release(_spec(args), cat, ws, args.lane, args.approved_by)
    print(gate_mod.to_markdown(out["decision"]))
    print(f"lanes: {out['promoted']}")
    if out["decision"]["decision"] != "approve":
        sys.exit(3)


def cmd_promote(args: argparse.Namespace) -> None:
    ws, _ = _ctx(args)
    try:
        state = Registry(ws.registry).promote(args.model, args.version, args.lane, args.approved_by, args.reason)
    except PromotionError as exc:
        raise SystemExit(f"refused: {exc}") from exc
    print(json.dumps({k: state[k] for k in ("dev", "staging", "prod")}, indent=2))


def cmd_rollback(args: argparse.Namespace) -> None:
    ws, _ = _ctx(args)
    state = Registry(ws.registry).rollback(args.model, args.lane, args.approved_by, args.reason)
    print(json.dumps({k: state[k] for k in ("dev", "staging", "prod")}, indent=2))


def cmd_online(args: argparse.Namespace) -> None:
    ws, cat = _ctx(args)
    groups = [g for g, v in cat.groups.items() if v.kind == "batch"]
    at = datetime.fromisoformat(args.at) if args.at else datetime.now()
    print(json.dumps(OnlineStore(ws.store / "online.sqlite").materialize(OfflineStore(ws.store), groups, at), indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from aucml.serving.app import create_app

    ws, _ = _ctx(args)
    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else unpack(Registry(ws.registry).package_path(args.model, args.lane), ws.root / ".serving" / args.model)
    )
    uvicorn.run(create_app(model_dir), host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aucml")
    p.add_argument("--root", default=".", help="workspace root (data/, store/, runs/, registry/)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("materialize", help="materialise batch feature groups into the offline store").set_defaults(fn=cmd_materialize)
    for name, fn, hlp in (("run", cmd_run, "run the 5-stage pipeline"), ("review", cmd_review, "pre-ship methodology review")):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("model")
        if name == "review":
            sp.add_argument("--run")
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("release", help="run, review, package, gate and promote")
    sp.add_argument("model")
    sp.add_argument("--lane", default="staging", choices=["dev", "staging", "prod"])
    sp.add_argument("--approved-by")
    sp.set_defaults(fn=cmd_release)
    sp = sub.add_parser("promote")
    sp.add_argument("model")
    sp.add_argument("version")
    sp.add_argument("lane", choices=["dev", "staging", "prod"])
    sp.add_argument("--approved-by")
    sp.add_argument("--reason", default="")
    sp.set_defaults(fn=cmd_promote)
    sp = sub.add_parser("rollback")
    sp.add_argument("model")
    sp.add_argument("lane", choices=["dev", "staging", "prod"])
    sp.add_argument("--approved-by", required=True)
    sp.add_argument("--reason", required=True)
    sp.set_defaults(fn=cmd_rollback)
    sp = sub.add_parser("online", help="load the online store as of a time")
    sp.add_argument("--at")
    sp.set_defaults(fn=cmd_online)
    sp = sub.add_parser("serve", help="serve a model over the SageMaker HTTP contract")
    sp.add_argument("model", nargs="?")
    sp.add_argument("--lane", default="prod")
    sp.add_argument("--model-dir")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8080)
    sp.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
