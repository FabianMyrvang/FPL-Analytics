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
# Lets this code find "seed_data/", "FPL_DATA/", "FPL-Core-Insights/" etc. whether it is
# run from notebooks/, scripts/, or the repo root.
import os
from pathlib import Path
if Path.cwd().name in ("notebooks", "scripts"):
    os.chdir(Path.cwd().parent)

# %%
# Data handling
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# Fuzzy matching
from thefuzz import fuzz, process
from rapidfuzz import process, fuzz

import git
import os
from time import sleep
from datetime import datetime
import json
import ast


import requests

# %%
players_df = pd.read_csv("seed_data/players.csv")
teams_df = pd.read_csv("seed_data/teams.csv")
positions_df = pd.read_csv("seed_data/positions.csv")
fixtures_df = pd.read_csv("seed_data/fixtures.csv")

fixtures_stats20_24 = pd.read_csv("seed_data/fixtures_stats.csv")
fpl20_24 = pd.read_csv("seed_data/fpl20_24.csv")

# %%
PLAYER_COLS = ['player_id','Full Name','Player Name','photo_url']

TEAM_COLS = ['team_id','team','team_badge']

POSITION_COLS = ['position_id', 'Position']

FIXTURES_COLS = ['match_id','gw_id','kickoff_time', 'home_team_id',
                       'away_team_id', 'home_team_name', 'away_team_name']

SEASON_COLS = ['gw_id','gw','gameweek','season']

FIXTURES_STATS_COLS  = ['match_id', 'gw_id', 'home_team_id', 'away_team_id', 'home_score',
       'away_score', 'team_h_difficulty', 'team_a_difficulty',
       'home_goals_scored', 'away_goals_scored', 'home_assists',
       'away_assists', 'home_own_goals', 'away_own_goals',
       'home_penalties_saved', 'away_penalties_saved', 'home_penalties_missed',
       'away_penalties_missed', 'home_yellow_cards', 'away_yellow_cards',
       'home_red_cards', 'away_red_cards', 'home_saves', 'away_saves',
       'home_bonus', 'away_bonus', 'home_bps', 'away_bps']

FPL_GAMEWEEK_COLS = ['match_id','gw_id', 'player_id', 'team_id', 'position_id', 'assists', 'bonus',
       'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored',
       'ict_index', 'influence', 'minutes', 'own_goals', 'penalties_missed',
       'penalties_saved', 'red_cards', 'saves', 'threat', 'total_points',
       'transfers_in', 'transfers_out', 'now_cost', 'yellow_cards',
       'expected_assists', 'expected_goal_involvements', 'expected_goals',
       'expected_goals_conceded', 'starts','clearances_blocks_interceptions', 
       'recoveries', 'tackles','defensive_contribution']


# %%
def get_current_season(today=None):
    """Return current football season as 'YYYY-YY' (e.g. '2025-26').
    The season runs Aug–May, so month >= 8 starts a new season."""
    d = today or datetime.now()
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


# %%
os.makedirs("Data", exist_ok=True)

BASE_URL = "https://fantasy.premierleague.com/api"

def fetch_json(endpoint):
    """Generic GET request to FPL API."""
    r = requests.get(f"{BASE_URL}{endpoint}")
    r.raise_for_status()
    return r.json()

# --- 1. Get bootstrap data ---
bootstrap = fetch_json("/bootstrap-static/")

# DIM: Players
fpl_api_players = pd.DataFrame(bootstrap['elements'])[
    ['id', 'code', 'photo','first_name', 'second_name', 'web_name',
     'team', 'element_type', 'now_cost',
     'selected_by_percent',  # selected
     'transfers_in',
     'transfers_out']
].rename(columns={'id': 'player_id', 'team': 'team_id', 'element_type': 'position_id'})

# Add full_name to fpl_api_players
fpl_api_players["full_name"] = fpl_api_players["first_name"] + " " + fpl_api_players["second_name"]

# DIM: Teams (include code for kit image URLs)
fpl_api_teams = pd.DataFrame(bootstrap['teams'])[
    ['id', 'code', 'name']
].rename(columns={'id': 'team_id', 'name': 'team'})

# Merge team kit code into players so creating_player_ids can build kit URLs
fpl_api_players = fpl_api_players.merge(
    fpl_api_teams[['team_id', 'code']].rename(columns={'code': 'team_code'}),
    on='team_id', how='left'
)

# DIM: Positions
fpl_api_positions = pd.DataFrame(bootstrap['element_types'])[
    ['id', 'singular_name']
].rename(columns={'id': 'position_id', 'singular_name': 'position_name'})

# DIM: Gameweeks
fpl_api_gameweeks = pd.DataFrame(bootstrap['events'])[
    ['id', 'name', 'deadline_time', 'average_entry_score', 'highest_score', 'finished']
].rename(columns={'id': 'gw', 'average_entry_score': 'average_score','name':'gameweek'})

# --- 2. DIM: Fixtures ---
fpl_api_fixtures = pd.DataFrame(fetch_json("/fixtures/"))[
    ['id', 'event', 'team_h', 'team_a', 'kickoff_time', 
     'team_h_score', 'team_a_score', 'team_h_difficulty', 'team_a_difficulty', 'stats']
].rename(columns={
    'id': 'fixture_id',
    'event': 'gw',
    'team_h': 'home_team_id',
    'team_a': 'away_team_id',
    'team_h_score': 'home_score',
    'team_a_score': 'away_score',
})

# --- 3. FACT: PlayerGameweekStats ---
finished_gws = fpl_api_gameweeks[fpl_api_gameweeks['finished'] == True]['gw'].tolist()
gws = []

for gw in finished_gws:
    print(f"Fetching GW{gw} data...")
    gw_data = fetch_json(f"/event/{gw}/live/")
    for el in gw_data['elements']:
        stats = el['stats']
        stats['player_id'] = el['id']
        stats['gw'] = gw
        gws.append(stats)
    sleep(2)  # avoid hammering the API

gws_df = pd.DataFrame(gws)

# %%
# --- 4. Create Master Player Stats Table ---

# Step 1: Prepare fixtures so each team appears once per fixture
fixtures_for_merge = pd.concat([
    fpl_api_fixtures.assign(player_team_id=fpl_api_fixtures['home_team_id'], is_home=True),
    fpl_api_fixtures.assign(player_team_id=fpl_api_fixtures['away_team_id'], is_home=False)
], ignore_index=True)

# Step 2: Create the master table
master_player_stats_table = (
    gws_df
    .merge(fpl_api_players, on='player_id', how='left', suffixes=('_gw_stat', '_player'))
    .merge(fpl_api_teams, on='team_id', how='left', suffixes=('', '_player_team'))
    .merge(fpl_api_positions, on='position_id', how='left')
    .merge(fpl_api_gameweeks, on='gw', how='left', suffixes=('', '_gameweek'))
    # Match player to their fixture based on gw AND team_id
    .merge(
        fixtures_for_merge,
        left_on=['gw', 'team_id'],
        right_on=['gw', 'player_team_id'],
        how='left'
    )
    # Add home team names
    .merge(
        fpl_api_teams[['team_id', 'team']].rename(columns={'team': 'home_team_name', 'team_id': 'home_team_id'}),
        on='home_team_id',
        how='left'
    )
    # Add away team names
    .merge(
        fpl_api_teams[['team_id', 'team']].rename(columns={'team': 'away_team_name', 'team_id': 'away_team_id'}),
        on='away_team_id',
        how='left'
    )
)


# %%
# --- 5. Create Master Fixtures Table ---

master_fixtures_table = (
    fpl_api_fixtures
    # Add home team names
    .merge(
        fpl_api_teams[['team_id', 'team']].rename(columns={'team': 'home_team_name', 'team_id': 'home_team_id'}),
        on='home_team_id',
        how='left'
    )
    # Add away team names
    .merge(
        fpl_api_teams[['team_id', 'team']].rename(columns={'team': 'away_team_name', 'team_id': 'away_team_id'}),
        on='away_team_id',
        how='left'
    )
    # Add gameweek info
    .merge(
        fpl_api_gameweeks[['gw', 'gameweek', 'deadline_time', 'finished']],
        on='gw',
        how='left'
    )
)

# Result
master_fixtures_table['result'] = master_fixtures_table.apply(
    lambda row: 'H' if row['home_score'] > row['away_score']
                else 'D' if row['home_score'] == row['away_score']
                else 'A' if pd.notna(row['home_score']) and pd.notna(row['away_score'])
                else None,
    axis=1
)


# %%
# Player merge function. This merges the fpl_api_players data with the historic player data
def creating_player_ids(new_players : pd.DataFrame, old_players : pd.DataFrame):

    new_players = new_players.copy()
    
    new_players = new_players[["player_id",'code','team_code','full_name','first_name','second_name',"web_name"]].copy()
    
    # Add photo URL — team kit image (always current, updates on transfer)
    new_players['photo_url'] = new_players['team_code'].apply(
        lambda code: f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{int(code)}-110.png"
    )
    
    # Add full name column
    new_players['full_name'] = new_players['first_name'] + " " + new_players['second_name']
    
    # Prepare lists to store fuzzy match results
    matched_ids = []
    matched_names = []
    matched_scores = []
    
    # List of names to match against
    lookup_names = old_players['full_name'].tolist()
    
    # Loop through each full name in df and perform fuzzy matching
    for name in new_players['full_name']:
        match = process.extractOne(
            query=name,
            choices=lookup_names,
            scorer=fuzz.token_set_ratio,  # Use token set ratio for matching
            score_cutoff=90,              # Only accept matches above this score
            processor=None                # Do not preprocess the strings
        )
        
        if match:
            best_match, score = match[0], match[1]
            idx = lookup_names.index(best_match)  # Get index in df2
            matched_ids.append(old_players.iloc[idx]['player_id'])
            matched_names.append(best_match)
            matched_scores.append(score)
        else:
            matched_ids.append(None)
            matched_names.append(None)
            matched_scores.append(None)
    
    # Add the results to the original df
    new_players['matched_name'] = matched_names
    new_players['match_score'] = matched_scores
    new_players['player_id'] = matched_ids
    
    # -----------------------------
    # Assign new player_ids to unmatched players
    # -----------------------------
    max_id = int(old_players['player_id'].max() or 0)  # Current max player_id in df2
    new_players_mask = new_players['player_id'].isna()
    
    new_players.loc[new_players_mask, 'player_id'] = range(
        max_id + 1,
        max_id + 1 + new_players_mask.sum()
    )
    
    # Ensure player_id column is integer
    new_players['player_id'] = new_players['player_id'].astype('Int64')
    
    # Concatenating players data from season 2020 to 2024 with 2025
    all_players = pd.concat(
        [old_players, new_players[['player_id','full_name','web_name','photo_url']]],
        ignore_index=True
    ).drop_duplicates('full_name', keep='last' ,ignore_index=False)
    
    # Dropping duplicates but keeping last instance
    all_players = all_players.drop_duplicates('player_id',keep='last', ignore_index= False)
    
    # fill the Nas
    all_players['web_name'] = all_players['web_name'].fillna(all_players['full_name'].str.split().str[-1])

    # Count how many times each web_name appears
    web_counts = all_players['web_name'].value_counts()

# Default: use web_name
    all_players['player_name'] = all_players['web_name']

# Rows where web_name is duplicated
    mask = all_players['web_name'].map(web_counts).gt(1)

# For duplicated web_names, use "F. Lastname" from full_name
    all_players.loc[mask, 'player_name'] = (
        all_players.loc[mask, 'full_name']
        .astype(str)
        .str.strip()
        .str.split()
        .apply(lambda x: f"{x[0][0]}. {x[-1]}")
        )
    
    return all_players


# %%
def map_player_ids(stats_table, players_dim):
    
    """Map player IDs to stats table using full_name"""
    stats_table["player_id"] = stats_table["full_name"]\
        .map(dict(zip(players_dim["full_name"], players_dim["player_id"])))
    return stats_table


# %%
def team_merge(new_teams : pd.DataFrame, old_teams : pd.DataFrame):
    
    new_teams = new_teams.copy()

    new_teams['team_id'] = new_teams['team'].map(dict(zip(old_teams['team'], old_teams['team_id'])))
    
    # Find the current maximum player_id (ignoring NaN)
    max_id = int(new_teams['team_id'].max() or 0)
    
    # Create a mask for rows where player_id is NaN
    new_teams_mask = new_teams['team_id'].isna()
    
    # Assign new sequential IDs to these rows
    new_teams.loc[new_teams_mask, 'team_id'] = range(
        max_id + 1,
        max_id + 1 + new_teams_mask.sum()
    )
    
    # Concatenating the Vastav teams data to fpl api 25 season
    all_teams = pd.concat([
        old_teams,
        new_teams[~new_teams['team_id'].isin(old_teams['team_id'])]
    ], ignore_index=True)
    
    # team id as integer
    all_teams['team_id'] = all_teams['team_id'].astype('Int64')
    
    return all_teams


# %%
#NEW!

def map_team_ids(master_df: pd.DataFrame, all_teams: pd.DataFrame):

    master_df = master_df.copy()
    all_teams = all_teams.copy()

    master_df["season"] = get_current_season()

    # Mapping dictionary
    team_map = dict(zip(all_teams["team"], all_teams["team_id"]))

    # Optional player team id
    if "team" in master_df.columns:
        master_df["team_id"] = master_df["team"].map(team_map).astype("Int64")

    # Home / Away ids
    master_df["home_team_id"] = master_df["home_team_name"].map(team_map).astype("Int64")
    master_df["away_team_id"] = master_df["away_team_name"].map(team_map).astype("Int64")

    # Remove rows with missing teams
    master_df = master_df.dropna(subset=["home_team_id", "away_team_id"]).copy()

    # Convert to int
    master_df["home_team_id"] = master_df["home_team_id"].astype(int)
    master_df["away_team_id"] = master_df["away_team_id"].astype(int)

    # match_id
    master_df["match_id"] = (
        master_df["season"].str[:4]
        + master_df["home_team_id"].astype(str).str.zfill(2)
        + master_df["away_team_id"].astype(str).str.zfill(2)
    ).astype("Int64")

    # gw_id
    season_year = master_df["season"].str[:4].astype("Int64")
    master_df["gw_id"] = season_year * 100 + master_df["gw"].astype("Int64")

    return master_df


# %%
def concat_gameweek_stats(master_gw_df, prev_seasons_gw, columns):
    """Concatenate current and historical gameweek stats, cleaning and reordering columns."""
    
    fpl_gw_current = master_gw_df.copy()
    fpl_gw_historic = prev_seasons_gw.copy()
    
    # Concatenate historical and current data
    fpl_gw_complete = pd.concat([fpl_gw_historic, fpl_gw_current], ignore_index=True)
    
    fpl_gw_complete = fpl_gw_complete[columns]
    
    return fpl_gw_complete


def upsert_stats(master_df, output_path, historical_df, columns, key_cols):
    """Upsert fresh current-season rows into the existing output CSV.

    The existing CSV (all prior seasons) is the base; the freshly fetched
    current-season rows win on key collision (keep='last'). This makes the
    pipeline non-destructive across season boundaries: re-running in a new
    season appends the new season instead of overwriting prior ones.
    The historical file is only used to seed the very first run, before any
    output CSV exists.
    """
    current = master_df.copy()
    base = pd.read_csv(output_path) if os.path.exists(output_path) else historical_df.copy()
    combined = pd.concat([base, current], ignore_index=True)[columns]
    return combined.drop_duplicates(subset=key_cols, keep="last")


# %%
def create_fixtures_stat_table(fixtures):

    fixtures = fixtures.copy()
    """
    Flatten the 'stats' column from FPL fixtures into separate home/away columns.
    Handles NaN, lists, strings, arrays, and malformed entries safely.
    """

    def parse_stats(x):
        # Only treat scalar NaN as empty
        if isinstance(x, float) and pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(x)
                except:
                    return []
        return []

    # Apply parsing to the stats column
    fixtures['stats'] = fixtures['stats'].apply(parse_stats)

    stats_records = []
    for _, row in fixtures.iterrows():
        stats_list = row['stats']

        match_stats = {
            'match_id': row.get('match_id'),
            'gw_id': row.get('gw_id'),
            'home_team_id': row.get('home_team_id'),
            'away_team_id': row.get('away_team_id'),
            'home_score': row.get('home_score'),
            'away_score': row.get('away_score'),
            'team_h_difficulty': row.get('team_h_difficulty'),
            'team_a_difficulty': row.get('team_a_difficulty')
        }

        # Flatten stats
        for stat in stats_list:
            if not isinstance(stat, dict):
                continue
            identifier = stat.get('identifier')
            if not identifier:
                continue
            match_stats[f'home_{identifier}'] = sum(s.get('value', 0) for s in stat.get('h', []))
            match_stats[f'away_{identifier}'] = sum(s.get('value', 0) for s in stat.get('a', []))

        stats_records.append(match_stats)

    return pd.DataFrame(stats_records)


# %%
# Position table
position_dim = positions_df

# Players lookup table
# Use the existing saved lookup as the base so previously-assigned player_ids
# (incl. current-season debutants) stay stable across seasons. Fall back to the
# static historical file only on the very first run.
player_base = pd.read_csv("FPL_DATA/player_dim.csv") if os.path.exists("FPL_DATA/player_dim.csv") else players_df
player_dim = creating_player_ids(fpl_api_players, player_base)

# Team lookup table (existing saved lookup as base for the same reason)
team_base = pd.read_csv("FPL_DATA/team_dim.csv") if os.path.exists("FPL_DATA/team_dim.csv") else teams_df
team_dim = team_merge(fpl_api_teams, team_base)

# MASTER PLAYER STATS TABLE
# Mapping player ids to the master table
master_player_stats_table = map_player_ids(master_player_stats_table, player_dim)

# Mapping new team ids to the master table
master_player_stats_table = map_team_ids(master_player_stats_table, team_dim)

# MASTER FIXTURE TABLE
master_fixtures_table = map_team_ids(master_fixtures_table, team_dim)

# TIME-SERIES FACT TABLES — upsert into existing output to preserve prior seasons.
# Fresh current-season rows win on key collision (keep='last'); this also subsumes
# the double-GW deduplication on (gw_id, player_id).
fpl_gameweek_fact = upsert_stats(master_player_stats_table, "FPL_DATA/fpl_gameweek_fact.csv", fpl20_24,            FPL_GAMEWEEK_COLS,   ['gw_id', 'player_id'])
fixture_dim     = upsert_stats(master_fixtures_table,     "FPL_DATA/fixture_dim.csv",     fixtures_df,         FIXTURES_COLS,       ['match_id'])
season_dim      = upsert_stats(master_fixtures_table,     "FPL_DATA/season_dim.csv",      fixtures_df,         SEASON_COLS,         ['gw_id']).dropna(subset=['gw_id']).reset_index(drop=True)
fpl_fixture_fact  = upsert_stats(master_fixtures_table,     "FPL_DATA/fpl_fixture_fact.csv",  fixtures_stats20_24, FIXTURES_STATS_COLS, ['match_id'])

# %%
# Export each DataFrame
player_dim.to_csv("FPL_DATA/player_dim.csv", index=False)
position_dim.to_csv("FPL_DATA/position_dim.csv", index = False)
team_dim.to_csv("FPL_DATA/team_dim.csv", index=False)
fixture_dim.to_csv("FPL_DATA/fixture_dim.csv", index=False)
season_dim.to_csv("FPL_DATA/season_dim.csv", index=False)

fpl_fixture_fact.to_csv("FPL_DATA/fpl_fixture_fact.csv",index= False)
fpl_gameweek_fact.to_csv("FPL_DATA/fpl_gameweek_fact.csv", index=False)

# %%
"""
def main():
    global master_player_stats_table, master_fixtures_table
    
    player_dim = creating_player_ids(fpl_api_players, players_df)
    team_dim = team_merge(fpl_api_teams, teams_df)
    position_dim = positions_df
    
    master_player_stats_table = map_player_ids(master_player_stats_table, player_dim)
    master_player_stats_table = team_transformer(master_player_stats_table, fpl_api_teams, teams_df)
    
    master_fixtures_table = team_transformer(master_fixtures_table, fpl_api_teams, teams_df)
    
    fpl_gameweek_fact = concat_gameweek_stats(master_player_stats_table, fpl20_24, FPL_GAMEWEEK_COLS)
    fixture_dim = concat_gameweek_stats(master_fixtures_table, fixtures_df, FIXTURES_COLS)
    season_dim = concat_gameweek_stats(master_fixtures_table, fixtures_df, SEASON_COLS).drop_duplicates().reset_index(drop=True)
    fpl_fixture_fact = concat_gameweek_stats(master_fixtures_table, fixtures_stats20_24, FIXTURES_STATS_COLS)
    
    player_dim.to_csv("FPL_DATA/player_dim.csv", index=False)
    position_dim.to_csv("FPL_DATA/position_dim.csv", index=False)
    team_dim.to_csv("FPL_DATA/team_dim.csv", index=False)
    fixture_dim.to_csv("FPL_DATA/fixture_dim.csv", index=False)
    season_dim.to_csv("FPL_DATA/season_dim.csv", index=False)
    fpl_fixture_fact.to_csv("FPL_DATA/fpl_fixture_fact.csv", index=False)
    fpl_gameweek_fact.to_csv("FPL_DATA/fpl_gameweek_fact.csv", index=False)
    
    print("Pipeline completed")

if __name__ == "__main__":
    main()
"""
