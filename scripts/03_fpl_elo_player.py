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
# run from notebooks/, scripts/, or the repo root.
import os
from pathlib import Path
if Path.cwd().name in ("notebooks", "scripts"):
    os.chdir(Path.cwd().parent)

# %%
# Data handling
import pandas as pd
import numpy as np

import git
import os
from datetime import datetime


# --- season awareness: derive the current season from today's date ---
# Mirrors 02_fpl_api.get_current_season(). A season that starts in August YYYY is
# labelled "YYYY-YY" (short, used in match_id) and stored by FPL-Core-Insights
# under the folder "YYYY-YYYY".
def get_current_season(today=None):
    today = today or datetime.now()
    start_year = today.year if today.month >= 8 else today.year - 1
    season_short  = f"{start_year}-{str(start_year + 1)[2:]}"   # e.g. "2025-26"
    season_folder = f"{start_year}-{start_year + 1}"            # e.g. "2025-2026"
    return season_short, season_folder


SEASON_SHORT, SEASON_FOLDER = get_current_season()
ELO_DATA_DIR = f"FPL-Core-Insights/data/{SEASON_FOLDER}"
print(f"Season: {SEASON_SHORT}  |  FPL-Core-Insights dir: {ELO_DATA_DIR}")


# --- non-destructive upsert: accumulate seasons instead of overwriting ---
# match_id is season-unique (prefixed with the season's start year), so older
# seasons are preserved and the current season's rows win on a key collision.
def upsert_csv(new_df, path, keys):
    if os.path.exists(path):
        base = pd.read_csv(path)
        combined = pd.concat([base, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    return combined

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


ELO_PLAYER_COLS = ['match_id','player_id','team_id', 'position_id', 'gw_id','was_home',
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
# Pull latest FPL-Core-Insights data (repo must be cloned at the project root)
git_dir = "FPL-Core-Insights"
g = git.cmd.Git(git_dir)
g.pull()

# %% [markdown]
# ### FPL ELO GAMEWEEK STATS

# %%
elo_players = pd.read_csv(f"{ELO_DATA_DIR}/players.csv")
elo_teams = pd.read_csv(f"{ELO_DATA_DIR}/teams.csv")
elo_path = f"{ELO_DATA_DIR}/By Tournament/Premier League/"


elo_fixtures_25 = []

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

        # Drop empty columns
        df = df.dropna(axis=1, how='all')

        elo_fixtures_25.append(df)
    else:
        break


elo_fixtures_25 = pd.concat(elo_fixtures_25, ignore_index=True)


# %%
# Map the home_team_id from fpl_elo_fixtures_25 and get the home_team_name from the elo_teams df
elo_fixtures_25 = pd.merge(elo_fixtures_25, elo_teams[["code","name"]], how = "left", left_on="home_team",right_on="code").drop(columns = ["code"]).rename(columns={"name":"home_team_name"})
# Map the away_team_id from fpl_elo_fixtures_25 and get the away_team_name from the elo_teams df
elo_fixtures_25 = pd.merge(elo_fixtures_25, elo_teams[["code","name"]], how = "left", left_on="away_team",right_on="code").drop(columns = ["code"]).rename(columns={"name":"away_team_name"})

#
elo_fixtures_25 = pd.merge(elo_fixtures_25,team_dim,how = "left", left_on="home_team_name", right_on="team").drop(columns=["team"]).rename(columns={"team_id":"home_team_id"})
elo_fixtures_25 = pd.merge(elo_fixtures_25,team_dim,how = "left", left_on="away_team_name", right_on="team").drop(columns=["team"]).rename(columns={"team_id":"away_team_id"})

# %%
# Match_id
elo_fixtures_25["season"] = SEASON_SHORT
elo_fixtures_25['match_id'] = (elo_fixtures_25['season'].str[:4] + elo_fixtures_25["home_team_id"].astype(str).str.zfill(2) + elo_fixtures_25['away_team_id'].astype(str).str.zfill(2)).astype('Int64')


# gw_id
# Convert match_id to string first, then extract the year
elo_fixtures_25['gw_id'] = elo_fixtures_25['match_id'].astype(str).str[:4].astype(int) * 100 + elo_fixtures_25['GW'].astype(int)

# %%
# Remove all columns where fpl_elo_fixtures has "pct" in its name
elo_fixtures_25 = elo_fixtures_25[[col for col in elo_fixtures_25.columns if 'pct' not in col]]

# Remove all columns where the NaN percentage is above 0.1
elo_fixtures_25 = elo_fixtures_25.loc[:, (elo_fixtures_25.isna().sum() / len(elo_fixtures_25)) <= 0.1]

# %%
upsert_csv(elo_fixtures_25, "FPL_DATA/elo_fixture_fact.csv", keys=["match_id"])

# %% [markdown]
# ### ELO PLAYER GAMEWEEK STATS

# %%
elo_path = f"{ELO_DATA_DIR}/By Tournament/Premier League/"

elo_25 = []

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
        
        df = df.dropna(axis=1, how='all')
        elo_25.append(df)
    else:
        break

elo_25 = pd.concat(elo_25, ignore_index=True)

# %%
# Dropping columns that have all columns = 0
columns_to_drop = []
for col in elo_25.columns:
    if(elo_25[col] == 0).all():
        columns_to_drop.append(col)
# Drop them
elo_25 = elo_25.drop(columns=columns_to_drop)

# Drop columns i already have
elo_25 = elo_25.drop(columns = ["minutes_played","goals","assists","xg","xa","penalties_missed","tackles","goals_conceded",])

elo_players["full_name"] = elo_players["first_name"] + " " + elo_players["second_name"]

# %%
elo_25 = pd.merge(elo_25, elo_players[["player_id","full_name"]], how = "left", on="player_id").drop(columns=["player_id"])
elo_25 = pd.merge(elo_25, elo_teams[["code","name"]], how = "left", left_on="home_team",right_on="code").drop(columns = ["code"]).rename(columns={"name":"home_team_name"})
elo_25 = pd.merge(elo_25, elo_teams[["code","name"]], how = "left", left_on="away_team",right_on="code").drop(columns = ["code"]).rename(columns={"name":"away_team_name"})
elo_25 = pd.merge(elo_25, team_dim, how = "left", left_on="home_team_name", right_on="team").drop(columns = ["team"]).rename(columns = {"team_id":"home_team_id"})
elo_25 = pd.merge(elo_25, team_dim, how = "left", left_on="away_team_name", right_on="team").drop(columns = ["team"]).rename(columns = {"team_id":"away_team_id"})

# %%
elo_25["season"] = SEASON_SHORT
elo_25['match_id'] = (elo_25['season'].str[:4] + elo_25["home_team_id"].astype(str).str.zfill(2) + elo_25['away_team_id'].astype(str).str.zfill(2)).astype('Int64')

# %%
# Getting the player id from the player dim table
elo_25 = pd.merge(elo_25, 
                      player_dim[["player_id","full_name"]], 
                      how = "left", 
                      left_on="full_name",
                      right_on= "full_name")

# getting team positi id
elo_25 = pd.merge(elo_25,
                      elo_players[["full_name","team_code","position"]],
                      how = "left",
                      on = "full_name")

# Getting team id
elo_25 = pd.merge(elo_25,
                      elo_teams[["code","name"]],
                      how = "left",
                      left_on= "team_code",
                      right_on="code").drop(columns=["team_code","code"])

elo_25 = pd.merge(elo_25,
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
elo_25["position"] = elo_25["position"].map(POS_FULL_TO_ABBR)

elo_25 = pd.merge(elo_25,
                      position_dim,
                      how = "left",
                      left_on= "position",
                      right_on="position").drop(columns=["position"])

# %%
elo_25['gw_id'] = elo_25['match_id'].astype(str).str[:4].astype(int) * 100 + elo_25['GW'].astype(int)

# generating a was_home binary column
elo_25["was_home"] = (elo_25["team_id"] == elo_25["home_team_id"]).astype(int)

# %%
elo_25 = elo_25[ELO_PLAYER_COLS]

# %%
upsert_csv(elo_25, "FPL_DATA/elo_gameweek_fact.csv", keys=["match_id", "player_id"])
