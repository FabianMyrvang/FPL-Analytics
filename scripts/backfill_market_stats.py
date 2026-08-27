"""One-off backfill of per-gameweek market stats into FPL_DATA/fpl_gameweek_fact.csv.

The FPL API only ever exposes *today's* ownership, price and transfer counts, so `02_fpl_api.py`
cannot reconstruct them for a season that has already been played — every gameweek row ends up
with the same value. FPL-Core-Insights archives the bootstrap snapshot per gameweek, which is the
only way to recover that history.

This patches ONLY the market columns of rows that already exist, matched on (gw_id, player_id).
Every other column, unmatched rows, and seasons not listed here are left untouched.

    python scripts/backfill_market_stats.py            # 2024-25 and 2025-26
    python scripts/backfill_market_stats.py --dry-run  # report without writing

Not part of the scheduled pipeline. From 2026-27 onwards `02_fpl_api.py` captures these figures
itself, freezing each gameweek on first write.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if Path.cwd().name in ("notebooks", "scripts"):
    os.chdir(Path.cwd().parent)

from common import player_name_key  # noqa: E402

FACT_PATH = "FPL_DATA/fpl_gameweek_fact.csv"
MARKET_COLS = ["selected_by_percent", "now_cost", "transfers_in", "transfers_out"]

# The repo's layout changed between seasons: 2024-25 keeps playerstats in a subfolder and omits
# the name columns, so it needs a separate players.csv to resolve ids.
# `cols` limits what each season is allowed to overwrite. 2020-24 came from the vaastav seed and
# already carries real per-gameweek price and transfer movement, so only the missing ownership is
# filled there. 2025-26 was built by this pipeline before the freeze existed and is flat across all
# 38 gameweeks, so every market column is repaired.
SEASONS = {
    2024: {
        "stats": "FPL-Core-Insights/data/2024-2025/playerstats/playerstats.csv",
        "names": "FPL-Core-Insights/data/2024-2025/players/players.csv",
        "cols": ["selected_by_percent"],
    },
    2025: {
        "stats": "FPL-Core-Insights/data/2025-2026/playerstats.csv",
        "names": None,  # first_name / second_name are inline
        "cols": MARKET_COLS,
    },
}


def resolve_by_token_subset(unresolved_keys, dim_keys):
    """Map source names onto fuller player_dim names, e.g. 'Marcos Senesi' -> 'Marcos Senesi Baron'.

    FPL-Core-Insights uses the short form while player_dim (seeded from the vaastav archive) often
    carries the full legal name. A plain folded-string match misses those.

    Requires one name's tokens to be a subset of the other's, and exactly one candidate to qualify.
    Deliberately stricter than fuzzy matching: FPL listed *managers* as assets in 2024-25, and a
    scorer would happily match 'Mikel Arteta' to 'Mikel Merino Zazon'. Subset matching cannot,
    because 'arteta' appears in no player name.

    Bidirectional, because the abbreviation runs either way: upstream has 'Marcos Senesi' where
    player_dim has 'Marcos Senesi Baron', but also 'Alisson Ramses Becker' where player_dim has the
    shorter 'Alisson Becker'. Both sides need at least two tokens, so a lone given name cannot
    swallow an unrelated player.
    """
    dim_tokens = {k: set(k.split()) for k in dim_keys}
    out = {}
    for key in unresolved_keys:
        tokens = set(key.split())
        if len(tokens) < 2:
            continue
        hits = [d for d, dt in dim_tokens.items()
                if len(dt) >= 2 and (tokens <= dt or dt <= tokens)]
        if len(hits) == 1:
            out[key] = hits[0]
    return out


def load_season(start_year, cfg, player_dim):
    """Return a frame of (gw_id, player_id, *MARKET_COLS) for one season, or None."""
    if not os.path.exists(cfg["stats"]):
        print(f"  {start_year}: {cfg['stats']} not found — skipping.")
        return None

    stats = pd.read_csv(cfg["stats"], low_memory=False)

    if cfg["names"]:
        names = pd.read_csv(cfg["names"])[["player_id", "first_name", "second_name"]]
        stats = stats.merge(names, left_on="id", right_on="player_id", how="left")

    missing = [c for c in MARKET_COLS + ["gw", "first_name", "second_name"]
               if c not in stats.columns]
    if missing:
        print(f"  {start_year}: source is missing {missing} — skipping.")
        return None

    stats["_name_key"] = player_name_key(stats["first_name"] + " " + stats["second_name"])
    stats = stats.merge(player_dim, on="_name_key", how="left")

    still = sorted(stats.loc[stats["dim_player_id"].isna(), "_name_key"].dropna().unique())
    if still:
        remap = resolve_by_token_subset(still, set(player_dim["_name_key"]))
        if remap:
            fixed = stats["_name_key"].map(remap)
            lookup = player_dim.set_index("_name_key")["dim_player_id"]
            stats["dim_player_id"] = stats["dim_player_id"].fillna(fixed.map(lookup))
            print(f"  {start_year}: matched {len(remap)} player(s) on fuller names "
                  f"(e.g. {list(remap.items())[0][0]!r} -> {list(remap.items())[0][1]!r})")
        left = sorted(set(still) - set(remap))
        if left:
            print(f"  {start_year}: {len(left)} name(s) still unresolved "
                  f"(managers are expected here): {left[:4]}")

    out = stats.dropna(subset=["dim_player_id"]).copy()
    out["player_id"] = out["dim_player_id"].astype(int)
    out["gw_id"] = start_year * 100 + out["gw"].astype(int)

    # Upstream stores price in millions (4.5); fpl_gameweek_fact uses the FPL API's tenths (45).
    out["now_cost"] = (out["now_cost"].astype(float) * 10).round().astype("Int64")

    # Blank out columns this season is not allowed to overwrite; the merge below only applies
    # non-null values, so they pass through untouched.
    for col in MARKET_COLS:
        if col not in cfg["cols"]:
            out[col] = pd.NA

    out = out[["gw_id", "player_id"] + MARKET_COLS]

    # A player can appear twice for one gameweek if two source rows fold onto the same
    # persistent id; last wins, matching the pipeline's upsert convention.
    return out.drop_duplicates(subset=["gw_id", "player_id"], keep="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    fact = pd.read_csv(FACT_PATH)
    print(f"{FACT_PATH}: {len(fact):,} rows, {len(fact.columns)} columns")

    if "selected_by_percent" not in fact.columns:
        # Insert next to the other market columns rather than at the end, so the schema reads
        # sensibly; position is cosmetic since consumers select by name.
        fact["selected_by_percent"] = pd.NA
        print("  added column: selected_by_percent")

    pdim = pd.read_csv("FPL_DATA/player_dim.csv")[["player_id", "full_name"]]
    pdim["_name_key"] = player_name_key(pdim["full_name"])
    pdim = pdim.rename(columns={"player_id": "dim_player_id"})[["dim_player_id", "_name_key"]]

    # Two different players can fold to the same key. Drop both rather than guess — a wrong
    # attribution is worse than a missing one, and it keeps the merge one-to-many.
    ambiguous = pdim["_name_key"].duplicated(keep=False)
    if ambiguous.any():
        print(f"  {int(ambiguous.sum())} player_dim row(s) share a name key and are excluded")
        pdim = pdim[~ambiguous]

    updates = []
    for start_year, cfg in SEASONS.items():
        got = load_season(start_year, cfg, pdim)
        if got is not None:
            print(f"  {start_year}-{str(start_year + 1)[2:]}: {len(got):,} market rows "
                  f"across {got.gw_id.nunique()} gameweeks")
            updates.append(got)

    if not updates:
        print("Nothing to backfill.")
        return

    market = pd.concat(updates, ignore_index=True)

    before = fact[MARKET_COLS].notna().sum().to_dict()
    fact = fact.merge(market, on=["gw_id", "player_id"], how="left", suffixes=("", "_new"))

    changed = 0
    for col in MARKET_COLS:
        new = fact[f"{col}_new"]
        mask = new.notna()
        changed += int(mask.sum())
        fact.loc[mask, col] = new[mask]
    fact = fact.drop(columns=[f"{c}_new" for c in MARKET_COLS])

    print(f"\ncells updated: {changed:,}")
    for col in MARKET_COLS:
        print(f"  {col:<22} populated {before[col]:>7,} -> {int(fact[col].notna().sum()):>7,}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    fact.to_csv(FACT_PATH, index=False)
    print(f"\nWrote {FACT_PATH} ({len(fact):,} rows, {len(fact.columns)} columns)")


if __name__ == "__main__":
    main()
