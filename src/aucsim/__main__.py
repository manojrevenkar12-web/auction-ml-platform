"""python -m aucsim [--seed N] [--out data/warehouse.duckdb]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aucsim.market import write_warehouse

p = argparse.ArgumentParser(prog="aucsim")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--out", default="data/warehouse.duckdb")
a = p.parse_args()
print(json.dumps(write_warehouse(Path(a.out), a.seed), indent=2))
