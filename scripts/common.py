"""Shared helpers for the FPL pipeline scripts.

Imported by 02_fpl_api.py, 03_fpl_elo_player.py and 04_fpl_elo_player_backup_scraper.py.
Kept dependency-light (pandas only) so it works under the slim requirements-ci.txt.
"""

import os
import unicodedata
from datetime import datetime

import pandas as pd


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


# Canonical team names live in FPL_DATA/team_dim.csv. Both the FPL API and
# FPL-Core-Insights occasionally rename a club mid-life — 2026-27 turned "Ipswich" into
# "Ipswich Town". Matching on the raw name would treat the rename as a debutant, mint a
# fresh team_id and orphan every prior season of that club's history. Worse, the elo
# scripts silently DROP rows whose team name will not resolve, so a rename can delete a
# club's fixtures rather than raise. Map incoming names onto the canonical form.
#
# Also covers the longer FBref spellings used by 04_fpl_elo_player_backup_scraper.py.
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


# Player names are the only key shared between FPL-Core-Insights and player_dim, and the two
# sources spell them differently. The FPL API also rewrites its own spellings from time to
# time — 2026-27 turned "Aarón Anselmino" into "Aaron Anselmino" and shortened
# "Mateus Gonçalo Espanha Fernandes" to "Mateus Fernandes" — while FPL-Core-Insights keeps the
# long form. Fold accents and case to absorb the cosmetic differences; PLAYER_ALIASES covers
# the ones folding cannot reach.
#
# The durable fix is to join on FPL's stable player `code` (published as `player_code` in
# FPL-Core-Insights' players.csv) rather than on names at all. That needs player_dim to carry
# the code, so it is a follow-up rather than a hotfix.
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

    combined.to_csv(path, index=False)
    print(f"Wrote {path}  ({len(combined)} rows, {len(combined.columns)} cols)")
    return combined
