"""Shared helpers for the FPL pipeline scripts.

Imported by 02_fpl_api.py, 03_fpl_detailed_player.py and 04_fpl_detailed_player_backup_scraper.py.
Kept dependency-light (pandas only) so it works under the slim requirements-ci.txt.
"""

import os
import unicodedata
from datetime import datetime

import pandas as pd


# Paths to every published output. Use these constants rather than literal paths: upsert_stats,
# upsert_csv and the player/team ID bases fall back to seed data when their file is missing, so a
# stale filename rebuilds history and reassigns IDs without raising.
DATA_DIR = "FPL_DATA"

DIM_PLAYER = f"{DATA_DIR}/dim_player.csv"
DIM_TEAM = f"{DATA_DIR}/dim_team.csv"
DIM_POSITION = f"{DATA_DIR}/dim_position.csv"
DIM_SEASON = f"{DATA_DIR}/dim_season.csv"
DIM_FIXTURE = f"{DATA_DIR}/dim_fixture.csv"

# Long format: one row per player per upcoming fixture, so it fans out 1:many against
# dim_player and is a fact/bridge rather than a dimension.
FACT_PLAYER_NEXT_FIXTURES = f"{DATA_DIR}/fact_player_next_fixtures.csv"
FACT_FPL_PLAYER_GW = f"{DATA_DIR}/fact_fpl_player_gw.csv"
FACT_FPL_FIXTURE = f"{DATA_DIR}/fact_fpl_fixture.csv"
FACT_DETAILED_PLAYER_GW = f"{DATA_DIR}/fact_detailed_player_gw.csv"
FACT_DETAILED_FIXTURE = f"{DATA_DIR}/fact_detailed_fixture.csv"


def get_current_season(today=None):
    """Return (season_short, season_folder) for the season containing `today`.

    A season starting in August YYYY is labelled "YYYY-YY" (short form, used as the
    match_id prefix) and stored by FPL-Core-Insights under the folder "YYYY-YYYY".
    Months before August belong to the season that started the previous year.

        >>> get_current_season(datetime(2026, 8, 20))
        ('2026-27', '2026-2027')
        >>> get_current_season(datetime(2026, 5, 20))
        ('2025-26', '2025-2026')

    Set FPL_SEASON_START_YEAR to pin the season explicitly — needed to backfill or
    re-verify a past season, since the upserts are keyed on a season-prefixed match_id
    and would otherwise only ever touch today's season.
    """
    override = os.environ.get("FPL_SEASON_START_YEAR")
    if override:
        start_year = int(override)
        print(f"FPL_SEASON_START_YEAR={start_year} — overriding the date-derived season.")
    else:
        today = today or datetime.now()
        start_year = today.year if today.month >= 8 else today.year - 1
    season_short = f"{start_year}-{str(start_year + 1)[2:]}"
    season_folder = f"{start_year}-{start_year + 1}"
    return season_short, season_folder


# Maps source spellings onto the canonical team names in FPL_DATA/dim_team.csv. Both upstream
# sources rename clubs occasionally (e.g. "Ipswich" -> "Ipswich Town"). An unmapped rename mints
# a fresh team_id and orphans that club's history, and the detailed scripts drop rows whose team
# will not resolve rather than raising. Also covers the longer FBref spellings used by 04_.
TEAM_ALIASES = {
    "Ipswich Town":      "Ipswich",
    "Leicester City":    "Leicester",
    "Leeds United":      "Leeds",
    "Manchester City":   "Man City",
    "Manchester Utd":    "Man Utd",
    "Newcastle United":  "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Spurs",
    "West Ham United":   "West Ham",
}


def normalise_team_names(df, col, known_teams=None, label=""):
    """Map `col` onto canonical team names in-place on a copy, and return it.

    If `known_teams` is given (typically team_dim['team']), any name that still fails to
    match after aliasing is reported — that is the early warning for the next rename.
    """
    df = df.copy()
    df[col] = df[col].map(TEAM_ALIASES).fillna(df[col])

    if known_teams is not None:
        unknown = sorted(set(df[col].dropna()) - set(known_teams))
        if unknown:
            where = f" in {label}" if label else ""
            print(f"WARNING — team names{where} not present in team_dim: {unknown}. "
                  "If any is a rename of an existing club, add it to TEAM_ALIASES; "
                  "otherwise it is a genuine debutant.")
    return df


# Player name is the only key shared between FPL-Core-Insights and dim_player, and the two
# sources spell it differently (accents, and long vs short forms). Folding accents and case
# absorbs the cosmetic differences; PLAYER_ALIASES covers what folding cannot reach.
#
# TODO: join on FPL's stable player `code` instead of names. FPL-Core-Insights publishes it as
# `player_code`; this needs dim_player to carry the code first.
PLAYER_ALIASES = {
    "Julio Enciso Espínola":            "Julio Enciso",
    "Mateus Gonçalo Espanha Fernandes": "Mateus Fernandes",
}


def fold_name(value):
    """Accent- and case-insensitive key for matching player names across sources."""
    if not isinstance(value, str):
        return value
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def player_name_key(series):
    """Map a Series of player names onto the canonical match key."""
    return series.map(PLAYER_ALIASES).fillna(series).map(fold_name)


def upsert_csv(new_df, path, keys, columns=None):
    """Upsert `new_df` into the CSV at `path`, keyed on `keys`.

    The existing file (all prior seasons) is the base and fresh rows win on a key
    collision, which makes re-runs idempotent and keeps the pipeline non-destructive
    across season boundaries. Writing is skipped entirely when there is nothing new:
    concatenating an empty frame would upcast int64 columns to float64 and rewrite the
    whole file as 0 -> 0.0 for no gain.

    `columns` pins the written schema. Without it the output is the union of the existing
    file and the new rows, so the column set drifts as upstream data fills in.
    """
    if new_df is None or new_df.empty:
        print(f"No new rows for {path} — leaving it untouched.")
        return pd.read_csv(path) if os.path.exists(path) else new_df

    if os.path.exists(path):
        base = pd.read_csv(path)
        combined = pd.concat([base, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    else:
        combined = new_df

    if columns is not None:
        combined = combined.reindex(columns=columns)

    # Sort before writing so the output is reproducible from the same inputs. Without this,
    # drop_duplicates(keep="last") leaves rows wherever the concat put them, so row order varies
    # between runs and every run produces a git diff with no value changes. mergesort is stable,
    # so equal keys keep their relative order.
    combined = combined.sort_values(keys, kind="mergesort").reset_index(drop=True)

    combined.to_csv(path, index=False)
    print(f"Wrote {path}  ({len(combined)} rows, {len(combined.columns)} cols)")
    return combined
