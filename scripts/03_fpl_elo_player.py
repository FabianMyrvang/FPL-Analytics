# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# --- repo-root bootstrap: resolve paths relative to the project root ---
# Lets this code find "FPL_DATA/", "FPL-Core-Insights/" etc. whether it is
# run from notebooks/, scripts/, or the repo root. We chdir away from scripts/, so put it
# on sys.path explicitly to keep `import common` working from either location.
import os
import sys
from pathlib import Path
_SCRIPTS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if Path.cwd().name in ("notebooks", "scripts"):
    os.chdir(Path.cwd().parent)

# %%
# Data handling
import pandas as pd
import numpy as np

import git
import os

from common import get_current_season, normalise_team_names, player_name_key, upsert_csv

SEASON_SHORT, SEASON_FOLDER = get_current_season()
ELO_DATA_DIR = f"FPL-Core-Insights/data/{SEASON_FOLDER}"
print(f"Season: {SEASON_SHORT}  |  FPL-Core-Insights dir: {ELO_DATA_DIR}")

# %%
# Read each CSV file
player_dim = pd.read_csv("FPL_DATA/player_dim.csv")
position_dim = pd.read_csv("FPL_DATA/position_dim.csv")
team_dim = pd.read_csv("FPL_DATA/team_dim.csv")
fixture_dim = pd.read_csv("FPL_DATA/fixture_dim.csv")
season_dim = pd.read_csv("FPL_DATA/season_dim.csv")

# %%
ELO_FIXTURES_COLS = ['match_id','gw_id','home_team_id', 'away_team_id','home_team_elo', 'away_team_elo', 'home_possession',
       'away_possession', 'home_expected_goals_xg', 'away_expected_goals_xg',
       'home_total_shots', 'away_total_shots', 'home_shots_on_target',
       'away_shots_on_target', 'home_big_chances', 'away_big_chances',
       'home_big_chances_missed', 'away_big_chances_missed',
       'home_accurate_passes', 'away_accurate_passes', 'home_fouls_committed',
       'away_fouls_committed', 'home_corners', 'away_corners',
       'home_xg_open_play', 'away_xg_open_play', 'home_xg_set_play',
       'away_xg_set_play', 'home_non_penalty_xg', 'away_non_penalty_xg',
       'home_xg_on_target_xgot', 'away_xg_on_target_xgot',
       'home_shots_off_target', 'away_shots_off_target', 'home_blocked_shots',
       'away_blocked_shots', 'home_hit_woodwork', 'away_hit_woodwork',
       'home_shots_inside_box', 'away_shots_inside_box',
       'home_shots_outside_box', 'away_shots_outside_box', 'home_passes',
       'away_passes', 'home_own_half', 'away_own_half', 'home_opposition_half',
       'away_opposition_half', 'home_accurate_long_balls',
       'away_accurate_long_balls', 'home_accurate_crosses',
       'away_accurate_crosses', 'home_throws', 'away_throws',
       'home_touches_in_opposition_box', 'away_touches_in_opposition_box',
       'home_offsides', 'away_offsides', 'home_yellow_cards',
       'away_yellow_cards', 'home_red_cards', 'away_red_cards',
       'home_tackles_won', 'away_tackles_won', 'home_interceptions',
       'away_interceptions', 'home_blocks', 'away_blocks', 'home_clearances',
       'away_clearances', 'home_keeper_saves', 'away_keeper_saves',
       'home_duels_won', 'away_duels_won', 'home_ground_duels_won',
       'away_ground_duels_won', 'home_aerial_duels_won',
       'away_aerial_duels_won', 'home_successful_dribbles',
       'away_successful_dribbles']

# The written schema for elo_fixture_fact.csv: the stat columns above plus the descriptive
# ones worth keeping. Deliberately excludes upstream internals that carry no analytical
# value — home_team/away_team (raw source codes, already resolved to *_team_id),
# match_url, fotmob_id, stats_processed, player_stats_processed, tournament, and GW
# (duplicates `gameweek`, and gw_id already encodes it).
ELO_FIXTURE_OUTPUT_COLS = ELO_FIXTURES_COLS + [
    'gameweek', 'home_score', 'away_score', 'finished',
    'home_team_name', 'away_team_name', 'season',
]


ELO_PLAYER_COLS =['match_id','player_id','team_id', 'position_id', 'gw_id','was_home',
                   'total_shots', 'shots_on_target', 'successful_dribbles',
                   'big_chances_missed', 'touches_opposition_box', 'touches',
                   'accurate_passes', 'chances_created', 'final_third_passes',
                   'accurate_crosses', 'accurate_long_balls', 'interceptions',
                   'recoveries', 'blocks', 'clearances', 'headed_clearances',
                   'dribbled_past', 'duels_won', 'duels_lost', 'ground_duels_won',
                   'aerial_duels_won', 'was_fouled', 'fouls_committed', 'saves',
                   'xgot_faced', 'goals_prevented', 'sweeper_actions',
                   'gk_accurate_passes', 'gk_accurate_long_balls', 'high_claim',
                   'offsides', 'xgot', 'start_min', 'finish_min', 'team_goals_conceded',
                   'penalties_scored' ]

# %%
# Pull latest FPL-Core-Insights data (repo must be cloned at the project root).
# Don't let a pull failure kill the run: CI clones the repo fresh so the pull is redundant
# there, and locally a dirty working tree in that mirror would otherwise abort the whole
# pipeline. Warn and carry on with whatever data is already on disk.
git_dir = "FPL-Core-Insights"
try:
    git.cmd.Git(git_dir).pull()
except Exception as exc:
    print(f"WARNING — could not pull {git_dir} ({exc.__class__.__name__}); "
          "continuing with the data already on disk. "
          "If it looks stale, check for local modifications in that clone.")

# %% [markdown]
# ### ELO FIXTURE (TEAM) STATS  ->  FPL_DATA/elo_fixture_fact.csv
#
# Per-match team statistics, read from each gameweek's `fixtures.csv`.
# The per-player table is built further down.

# %%
# FPL-Core-Insights only creates the season folder once it starts publishing. Exit cleanly
# rather than raising a bare FileNotFoundError from inside a read_csv.
if not os.path.isdir(ELO_DATA_DIR):
    print(f"{ELO_DATA_DIR} does not exist yet — FPL-Core-Insights has not published "
          f"{SEASON_FOLDER} data. Leaving existing FPL_DATA CSVs untouched.")
    sys.exit(0)

elo_players = pd.read_csv(f"{ELO_DATA_DIR}/players.csv")
elo_teams = pd.read_csv(f"{ELO_DATA_DIR}/teams.csv")

# Upstream uses the same club names as the FPL API ("Ipswich Town"), while team_dim's
# canonical form is "Ipswich". An unresolved name yields a NaN team_id, and those rows are
# then silently dropped — so normalise before any merge onto team_dim.
elo_teams = normalise_team_names(
    elo_teams, "name", team_dim["team"], label="FPL-Core-Insights teams.csv"
)

elo_path = f"{ELO_DATA_DIR}/By Tournament/Premier League/"


elo_fixtures = []

for gw in range(1, 39):
    gw_folder = os.path.join(elo_path, f"GW{gw}")
    file_path = os.path.join(gw_folder, "fixtures.csv")

    if os.path.exists(file_path):
        # Read fixtures
        df = pd.read_csv(file_path)
        df["GW"] = gw

        # Normalize 'finished' column to boolean if it exists
        if "finished" in df.columns:
            # Convert to string, strip whitespace, lowercase, then compare
            df = df[df["finished"].astype(str).str.strip().str.lower() == "true"]

        # Drop empty columns — but only when rows survived the `finished` filter above.
        # On a 0-row frame every column is trivially "all NaN", so this would drop the
        # entire schema and the merges below would fail with KeyError.
        if not df.empty:
            df = df.dropna(axis=1, how='all')

        elo_fixtures.append(df)
    else:
        break


# Every GW folder now exists from the start of a season, so the loop above reads all 38 and
# most yield zero finished fixtures early on. A 0-row frame still carries its columns, so the
# merges below are safe; only a completely empty list would break pd.concat.
if not elo_fixtures:
    print(f"No gameweek folders found under {elo_path} — nothing to process.")
    sys.exit(0)

elo_fixtures = pd.concat(elo_fixtures, ignore_index=True)
print(f"Finished fixtures found: {len(elo_fixtures)}")


# %%
# Resolve the home team: source team code -> team name (elo_teams) -> persistent team_id (team_dim)
elo_fixtures = pd.merge(elo_fixtures, elo_teams[["code","name"]], how = "left", left_on="home_team",right_on="code").drop(columns = ["code"]).rename(columns={"name":"home_team_name"})
# Same for the away team
elo_fixtures = pd.merge(elo_fixtures, elo_teams[["code","name"]], how = "left", left_on="away_team",right_on="code").drop(columns = ["code"]).rename(columns={"name":"away_team_name"})

#
elo_fixtures = pd.merge(elo_fixtures,team_dim,how = "left", left_on="home_team_name", right_on="team").drop(columns=["team"]).rename(columns={"team_id":"home_team_id"})
elo_fixtures = pd.merge(elo_fixtures,team_dim,how = "left", left_on="away_team_name", right_on="team").drop(columns=["team"]).rename(columns={"team_id":"away_team_id"})

# %%
# Match_id
elo_fixtures["season"] = SEASON_SHORT
elo_fixtures['match_id'] = (elo_fixtures['season'].str[:4] + elo_fixtures["home_team_id"].astype(str).str.zfill(2) + elo_fixtures['away_team_id'].astype(str).str.zfill(2)).astype('Int64')


# gw_id
# Convert match_id to string first, then extract the year
elo_fixtures['gw_id'] = elo_fixtures['match_id'].astype(str).str[:4].astype(int) * 100 + elo_fixtures['GW'].astype(int)

# %%
# Pin the output schema. The previous ">10% NaN" filter made the column set data-dependent,
# so the file silently changed width as the season filled in (93 -> 95 during 2025-26) and
# broke consumers that assumed a fixed layout. It also divided by len(df), which is zero
# before any match is played. Reindexing is deterministic: absent columns arrive as NaN.
elo_fixtures = elo_fixtures.reindex(columns=ELO_FIXTURE_OUTPUT_COLS)

# %%
upsert_csv(elo_fixtures, "FPL_DATA/elo_fixture_fact.csv",
           keys=["match_id"], columns=ELO_FIXTURE_OUTPUT_COLS)

# %% [markdown]
# ### ELO PLAYER GAMEWEEK STATS

# %%
elo_path = f"{ELO_DATA_DIR}/By Tournament/Premier League/"

elo_player_match_stats = []

for gw in range(1, 39):
    gw_folder = os.path.join(elo_path, f"GW{gw}")
    file_path = os.path.join(gw_folder, "playermatchstats.csv")
    

    if os.path.exists(file_path):

        # Read player_files
        df = pd.read_csv(file_path)
        df["GW"] = gw

        # Read fixtures
        fixtures_file_path = os.path.join(gw_folder, "fixtures.csv")
        if os.path.exists(fixtures_file_path):
            fixtures = pd.read_csv(fixtures_file_path)
            fixtures = fixtures[['match_id', 'home_team', 'away_team']]
            df = df.merge(fixtures, on='match_id', how='left')
        
        # Same guard as the fixtures loop: a 0-row frame would lose every column here.
        if not df.empty:
            df = df.dropna(axis=1, how='all')
        elo_player_match_stats.append(df)
    else:
        break

if not elo_player_match_stats:
    print(f"No playermatchstats.csv files found under {elo_path}.")
    sys.exit(0)

elo_player_match_stats = pd.concat(elo_player_match_stats, ignore_index=True)

# Before the season starts these files are header-only. With zero rows every column
# trivially satisfies "all values == 0" and would be dropped below, after which the
# explicit drop list raises KeyError. There is nothing to upsert either way.
if elo_player_match_stats.empty:
    print("No player match stats published yet — leaving elo_gameweek_fact.csv untouched.")
    sys.exit(0)

print(f"Player match stat rows found: {len(elo_player_match_stats)}")

# %%
# Dropping columns that have all columns = 0
columns_to_drop = []
for col in elo_player_match_stats.columns:
    if(elo_player_match_stats[col] == 0).all():
        columns_to_drop.append(col)
# Drop them
elo_player_match_stats = elo_player_match_stats.drop(columns=columns_to_drop)

# Drop columns i already have. errors='ignore' because the all-zero pass above can legitimately
# have removed some of these early in a season, and an upstream column may disappear entirely.
elo_player_match_stats = elo_player_match_stats.drop(columns = ["minutes_played","goals","assists","xg","xa","penalties_missed","tackles","goals_conceded",], errors='ignore')

elo_players["full_name"] = elo_players["first_name"] + " " + elo_players["second_name"]

# %%
elo_player_match_stats = pd.merge(elo_player_match_stats, elo_players[["player_id","full_name"]], how = "left", on="player_id").drop(columns=["player_id"])
elo_player_match_stats = pd.merge(elo_player_match_stats, elo_teams[["code","name"]], how = "left", left_on="home_team",right_on="code").drop(columns = ["code"]).rename(columns={"name":"home_team_name"})
elo_player_match_stats = pd.merge(elo_player_match_stats, elo_teams[["code","name"]], how = "left", left_on="away_team",right_on="code").drop(columns = ["code"]).rename(columns={"name":"away_team_name"})
elo_player_match_stats = pd.merge(elo_player_match_stats, team_dim, how = "left", left_on="home_team_name", right_on="team").drop(columns = ["team"]).rename(columns = {"team_id":"home_team_id"})
elo_player_match_stats = pd.merge(elo_player_match_stats, team_dim, how = "left", left_on="away_team_name", right_on="team").drop(columns = ["team"]).rename(columns = {"team_id":"away_team_id"})

# %%
elo_player_match_stats["season"] = SEASON_SHORT
elo_player_match_stats['match_id'] = (elo_player_match_stats['season'].str[:4] + elo_player_match_stats["home_team_id"].astype(str).str.zfill(2) + elo_player_match_stats['away_team_id'].astype(str).str.zfill(2)).astype('Int64')

# %%
# Getting the player id from the player dim table.
# Join on a folded key rather than the raw name: FPL-Core-Insights and player_dim spell the
# same player differently, and the FPL API periodically rewrites its own spelling (2026-27
# dropped the accents from "Aarón Anselmino" and shortened several Portuguese names), which
# silently produced NaN player_ids. See PLAYER_ALIASES in common.py.
_pdim = player_dim[["player_id", "full_name"]].copy()
_pdim["_name_key"] = player_name_key(_pdim["full_name"])
_pdim = _pdim.drop(columns=["full_name"])

elo_player_match_stats["_name_key"] = player_name_key(elo_player_match_stats["full_name"])
elo_player_match_stats = pd.merge(elo_player_match_stats, _pdim, how="left", on="_name_key")

_missing = elo_player_match_stats.loc[elo_player_match_stats["player_id"].isna(), "full_name"].dropna().unique()
if len(_missing):
    print(f"WARNING — {len(_missing)} player name(s) did not match player_dim: "
          f"{sorted(_missing)[:10]}. Add them to PLAYER_ALIASES in common.py.")

elo_player_match_stats = elo_player_match_stats.drop(columns=["_name_key"])

# getting team positi id
elo_player_match_stats = pd.merge(elo_player_match_stats,
                      elo_players[["full_name","team_code","position"]],
                      how = "left",
                      on = "full_name")

# Getting team id
elo_player_match_stats = pd.merge(elo_player_match_stats,
                      elo_teams[["code","name"]],
                      how = "left",
                      left_on= "team_code",
                      right_on="code").drop(columns=["team_code","code"])

elo_player_match_stats = pd.merge(elo_player_match_stats,
                      team_dim,
                      how = "left",
                      left_on= "name",
                      right_on="team").drop(columns=["name","team"])

# Map full position names to abbreviations (position_dim uses GK/DEF/MID/FWD)
POS_FULL_TO_ABBR = {
    "Goalkeeper": "GK",
    "Defender":   "DEF",
    "Midfielder": "MID",
    "Forward":    "FWD",
}
elo_player_match_stats["position"] = elo_player_match_stats["position"].map(POS_FULL_TO_ABBR)

elo_player_match_stats = pd.merge(elo_player_match_stats,
                      position_dim,
                      how = "left",
                      left_on= "position",
                      right_on="position").drop(columns=["position"])

# %%
elo_player_match_stats['gw_id'] = elo_player_match_stats['match_id'].astype(str).str[:4].astype(int) * 100 + elo_player_match_stats['GW'].astype(int)

# generating a was_home binary column
elo_player_match_stats["was_home"] = (elo_player_match_stats["team_id"] == elo_player_match_stats["home_team_id"]).astype(int)

# %%
# reindex rather than strict selection: the all-zero pass above legitimately removes columns
# early in a season, and [ELO_PLAYER_COLS] would then raise KeyError. Missing ones become NaN.
elo_player_match_stats = elo_player_match_stats.reindex(columns=ELO_PLAYER_COLS)

# %%
upsert_csv(elo_player_match_stats, "FPL_DATA/elo_gameweek_fact.csv",
           keys=["match_id", "player_id"], columns=ELO_PLAYER_COLS)
