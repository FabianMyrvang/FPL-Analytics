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
# Lets this code find "seed_data/", "FPL_DATA/", "Fantasy-Premier-League/" etc. whether it is
# run from notebooks/, scripts/, or the repo root.
import os
from pathlib import Path
if Path.cwd().name in ("notebooks", "scripts"):
    os.chdir(Path.cwd().parent)

# %% [markdown]
# ### Fantasy premier league merging data

# %%
# Data handling
import pandas as pd
import numpy as np

import json
import ast

pd.set_option('display.max_columns', None)


# Fuzzy matching
from thefuzz import fuzz, process

import os

# Base URL for the vaastav Fantasy-Premier-League archive
VAASTAV = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Create seed_data/ output folder if it doesn't exist
os.makedirs("seed_data", exist_ok=True)


# %%
def fpl_merge_func(season20, teams20, season21, teams21, season22, teams22, season23, teams23):
    # Load all season data
    df_20 = pd.read_csv(season20).assign(season="2020-21")
    df_21 = pd.read_csv(season21).assign(season="2021-22")
    df_22 = pd.read_csv(season22).assign(season="2022-23")
    df_23 = pd.read_csv(season23).assign(season="2023-24")

    # Load all teams data
    teams20 = pd.read_csv(teams20)
    teams21 = pd.read_csv(teams21)
    teams22 = pd.read_csv(teams22)
    teams23 = pd.read_csv(teams23)

    # Merge only opponent team name with clean column naming
    df_20 = pd.merge(
        df_20,
        teams20[['id', 'name']].rename(columns={"name": "opp_team_name"}),
        how="left",
        left_on="opponent_team",
        right_on="id"
    ).drop(columns=["id"])

    df_21 = pd.merge(
        df_21,
        teams21[['id', 'name']].rename(columns={"name": "opp_team_name"}),
        how="left",
        left_on="opponent_team",
        right_on="id"
    ).drop(columns=["id"])

    df_22 = pd.merge(df_22,teams22[['id', 'name']].rename(columns={"name": "opp_team_name"}),
        how="left",
        left_on="opponent_team",
        right_on="id"
    ).drop(columns=["id"])

    df_23 = pd.merge(
        df_23,
        teams23[['id', 'name']].rename(columns={"name": "opp_team_name"}),
        how="left",
        left_on="opponent_team",
        right_on="id"
    ).drop(columns=["id"])

    # Combine all into one DataFrame
    fpl_all = pd.concat([df_20, df_21, df_22, df_23], ignore_index=True, sort=False)

    return fpl_all



# %%
fpl_merged_df = fpl_merge_func(
    season20= f'{VAASTAV}/2020-21/gws/merged_gw.csv',
    teams20=  f'{VAASTAV}/2020-21/teams.csv',

    season21= f'{VAASTAV}/2021-22/gws/merged_gw.csv',
    teams21=  f'{VAASTAV}/2021-22/teams.csv',

    season22= f'{VAASTAV}/2022-23/gws/merged_gw.csv',
    teams22=  f'{VAASTAV}/2022-23/teams.csv',

    season23= f'{VAASTAV}/2023-24/gws/merged_gw.csv',
    teams23=  f'{VAASTAV}/2023-24/teams.csv',
    )

# %%
# Since the season for 24-25 is not in the full_fpl dataset, we need to append this to the full dataset so that we have all the years from 2020-2024 season
# Looping through all the csv files and append into a complete 24-25 dataset with all gameweeks
teams_df = pd.read_csv(f'{VAASTAV}/2024-25/teams.csv')

# Empty list
season24_df = []

# Looping through gw 1-38 csv files and concatenating them to one full dataset for season 24-25
for gw in range(1, 39):
    df = pd.read_csv(f'{VAASTAV}/2024-25/gws/gw{gw}.csv')
    df["GW"] = gw
    season24_df.append(df)

fpl_24 = pd.concat(season24_df, ignore_index=True)

# Insert season
fpl_24.insert(0, "season", "2024-25")

# Lookup the opp_team_name from the teams_df and assign it to the fpl_24 df
fpl_24['opp_team_name'] = fpl_24['opponent_team'].map(dict(zip(teams_df['id'], teams_df['name'])))


# %%
# Keeping all columns from the fpl_full data
fpl_trimmed = fpl_24[fpl_merged_df.columns]

# Merging the fpl 2020-24 data with 24-25
fpl20_24 = pd.concat([fpl_merged_df, fpl_trimmed], ignore_index=True, sort=False)

# Drop element and round and opponent team columns
fpl20_24 = fpl20_24.drop(columns = ['element','round','opponent_team'])

# Dropping assistant managers
fpl20_24 = fpl20_24[fpl20_24['position'] != 'AM']

# Define mapping from short code / abbreviation to full name
position_mapping = {
    'GKP': 'GK',
    'GK': 'GK',
    'DEF': 'DEF',
    'MID': 'MID',
    'FWD': 'FWD'
}

# Replace all in your DataFrame
fpl20_24['position'] = fpl20_24['position'].replace(position_mapping)

# %%
# There are some differences in the names like for example Adama Traore and Adama Traore Diarra
# Therefore we need to do fuzzymatching with the fuzzymatch library
# But first I creata list of unique name

# Optional: remove leading/trailing spaces as well
fpl20_24['name'] = fpl20_24['name'].str.strip()

# Replace '-' with space in the 'name' column
fpl20_24['name'] = fpl20_24['name'].str.replace('-', ' ', regex=False)

# Unique name list
unique_names = fpl20_24['name'].dropna().unique().tolist()
len(unique_names)

# Empty dictionary
name_map = {}

# For every index in name column, iterate through each 
for i, name in enumerate(unique_names):

    # The name which will be returned by the fuzzy matching machine
    last_instance_name = name

    for fuzzy_name in unique_names[i+1:]:
        if fuzz.token_set_ratio(name, fuzzy_name)>=93: #Threshold, may adjust
            last_instance_name = fuzzy_name
    
    name_map[name] = last_instance_name


fpl20_24.insert(fpl20_24.columns.get_loc('name') + 1, 'full_name',fpl20_24['name'].replace(name_map))


# Creating unique id for player, team and season
fpl20_24 = fpl20_24.assign(
    player_id =  pd.factorize(fpl20_24["full_name"], sort=True)[0] + 1,
    team_id   = pd.factorize(fpl20_24["team"], sort=True)[0] + 1,
    position_id = fpl20_24["position"].map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4})
  )

# Drop regular name column
fpl20_24 = fpl20_24.drop(columns = ['name'])

# %%
# empty mapping list
name_matches = []

# Loop through all unique pairs
for i, name in enumerate(unique_names):
    for fuzzy_name in unique_names[i+1:]:
        score = fuzz.token_set_ratio(name, fuzzy_name)
        if score >= 93:  # similarity threshold
            name_matches.append((name, fuzzy_name, score))

# Convert to DataFrame
name_matches_df = pd.DataFrame(name_matches, columns=["name1", "name2", "similarity"])

# %%
cols = ['first_name', 'second_name', 'id', 'web_name']
web_names21 = pd.read_csv(f'{VAASTAV}/2020-21/players_raw.csv', usecols=cols)
web_names22 = pd.read_csv(f'{VAASTAV}/2021-22/players_raw.csv', usecols=cols)
web_names23 = pd.read_csv(f'{VAASTAV}/2022-23/players_raw.csv', usecols=cols)
web_names24 = pd.read_csv(f'{VAASTAV}/2023-24/players_raw.csv', usecols=cols)

web_names = pd.concat([web_names21, web_names22, web_names23, web_names24], ignore_index=True)
web_names['full_name'] = web_names['first_name'] + " " + web_names['second_name']
web_names['full_name'] = web_names['full_name'].str.replace("-", " ")

web_names = web_names.drop_duplicates("full_name", ignore_index= False)
fpl20_24 = pd.merge(fpl20_24, web_names[['full_name','first_name','second_name','web_name']], how = 'left', on = 'full_name')

# %%
players_df = fpl20_24[['player_id','full_name', 'web_name']].drop_duplicates().sort_values(by='player_id', ascending=True).reset_index(drop = True)
teams_df = fpl20_24[['team_id','team']].drop_duplicates().sort_values(by='team_id', ascending=True).reset_index(drop = True)
positions_df = fpl20_24[[ 'position_id', 'position']].drop_duplicates().sort_values(by='position_id', ascending=True).reset_index(drop = True)


# %%
def fpl_fixtures_func(fixtures20, teams20, fixtures21, teams21, fixtures22, teams22, fixtures23, teams23, fixtures24, teams24):
    # Load fixtures
    fix_20 = pd.read_csv(fixtures20).assign(season="2020-21")
    fix_21 = pd.read_csv(fixtures21).assign(season="2021-22")
    fix_22 = pd.read_csv(fixtures22).assign(season="2022-23")
    fix_23 = pd.read_csv(fixtures23).assign(season="2023-24")
    fix_24 = pd.read_csv(fixtures24).assign(season="2024-25")

    # Load teams
    teams20 = pd.read_csv(teams20)
    teams21 = pd.read_csv(teams21)
    teams22 = pd.read_csv(teams22)
    teams23 = pd.read_csv(teams23)
    teams24 = pd.read_csv(teams24)

    # Merge team names (home & away) with clean columns
    def merge_team_names(fix_df, teams_df):
        fix_df = fix_df.merge(
            teams_df[['id', 'name']].rename(columns={'name': 'home_team_name'}),
            left_on='team_h', right_on='id', how='left'
        )

        fix_df = fix_df.merge(
            teams_df[['id', 'name']].rename(columns={'name': 'away_team_name'}),
            left_on='team_a', right_on='id', how='left'
        )

        # Now drop all id columns
        fix_df = fix_df.drop(columns=[col for col in ['id_x', 'id_y','id'] if col in fix_df.columns])

        fix_df = fix_df.rename(columns={'event': 'gw'})

        return fix_df

    fix_20 = merge_team_names(fix_20, teams20)
    fix_21 = merge_team_names(fix_21, teams21)
    fix_22 = merge_team_names(fix_22, teams22)
    fix_23 = merge_team_names(fix_23, teams23)
    fix_24 = merge_team_names(fix_24, teams24)

    # Combine all
    fixtures_all = pd.concat([fix_20, fix_21, fix_22, fix_23, fix_24], ignore_index=True, sort=False)

    fixtures_all = fixtures_all.drop(columns = ['minutes','code','finished', 'finished_provisional', 'provisional_start_time','started','pulse_id'])
    return fixtures_all



# %%
fixtures = fpl_fixtures_func(
    fixtures20= f'{VAASTAV}/2020-21/fixtures.csv',
    teams20=    f'{VAASTAV}/2020-21/teams.csv',

    fixtures21= f'{VAASTAV}/2021-22/fixtures.csv',
    teams21=    f'{VAASTAV}/2021-22/teams.csv',

    fixtures22= f'{VAASTAV}/2022-23/fixtures.csv',
    teams22=    f'{VAASTAV}/2022-23/teams.csv',

    fixtures23= f'{VAASTAV}/2023-24/fixtures.csv',
    teams23=    f'{VAASTAV}/2023-24/teams.csv',

    fixtures24= f'{VAASTAV}/2024-25/fixtures.csv',
    teams24=    f'{VAASTAV}/2024-25/teams.csv',
    )

# %%

# gameweek ids
#fixtures['gw_id'] = fixtures['season'].str[:4].astype(int) * 100 + fixtures['gw'].astype(int)

fixtures['match_id'] = fixtures['season'] + "-" + fixtures['home_team_name'] + '-' + fixtures['away_team_name']

fixtures = pd.merge(
    fixtures, 
    teams_df[['team','team_id']], 
    left_on= 'home_team_name',
    right_on= 'team').drop(columns = ['team']).rename(columns = {'team_id':'home_team_id'})

fixtures = pd.merge(
    fixtures, 
    teams_df[['team','team_id']], 
    left_on= 'away_team_name',
    right_on= 'team').drop(columns = ['team']).rename(columns = {'team_id':'away_team_id'})

fixtures = fixtures.rename(columns= {'team_h_score':'home_score', 'team_a_score':'away_score'})

# Adjust gameweek column to have "Gameweek" in front of number
fixtures["gameweek"] = 'Gameweek' + ' ' + fixtures['gw'].astype(str)

# Overwriting the match_id cuz i needed to change it 
#fixtures['match_id'] = (fixtures['season'].str[1:4] + fixtures["home_team_id"].astype(str).str.zfill(2) + fixtures['away_team_id'].astype(str).str.zfill(2))

# adjusting gw column 
#
fixtures['gw_id'] = fixtures['match_id'].str[:4].astype('Int64') * 100 + fixtures['gw'].astype('Int64')


fixtures_df = fixtures[['match_id','home_team_id','away_team_id','home_team_name','away_team_name','home_score','away_score','team_h_difficulty','team_a_difficulty','gw','kickoff_time','season','stats','gameweek','gw_id']].drop_duplicates().reset_index(drop = True)

# %%
# Add match_id, if was_home is true then "team" comes first, if else then "opp_team_name" comes first
fpl20_24["match_id"] = (
    fpl20_24["season"] + "-" + 
    np.where(fpl20_24["was_home"], fpl20_24["team"], fpl20_24["opp_team_name"]) + "-" +
    np.where(fpl20_24["was_home"], fpl20_24["opp_team_name"], fpl20_24["team"])
)

# %%
# realized i needed home_team_id and away_team_id
fpl20_24 = pd.merge(fpl20_24, fixtures_df[["match_id","home_team_id","away_team_id"]], how = "left",on="match_id")

# Overwriting the match_id cuz i needed to change it 
fpl20_24['match_id'] = (fpl20_24['season'].str[:4] + fpl20_24["home_team_id"].astype(str).str.zfill(2) + fpl20_24['away_team_id'].astype(str).str.zfill(2)).astype('Int64')

 # adjusting gw column 
#fpl20_24['gw_id'] = fpl20_24['match_id'][:4].astype(int) * 100 + fpl20_24['GW'].astype(int)

#fpl20_24['gw_id'] = fixtures['match_id'].str[:4].astype(int) * 100 + fixtures['gw'].astype(int)

# gameweek ids
fpl20_24['gw_id'] = fpl20_24['season'].str[:4].astype('Int64') * 100 + fpl20_24['GW'].astype('Int64')

# %%
# Overwriting the match_id cuz i needed to change it 
fixtures['match_id'] = (fixtures['season'].str[:4] + fixtures["home_team_id"].astype(str).str.zfill(2) + fixtures['away_team_id'].astype(str).str.zfill(2)).astype('Int64')
fixtures_df['match_id'] = (fixtures_df['season'].str[:4] + fixtures_df["home_team_id"].astype(str).str.zfill(2) + fixtures_df['away_team_id'].astype(str).str.zfill(2)).astype('int64')


# %%
def parse_stats(x):
    # Handle various null-like values
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):  # Already a list
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(x)  # Try ast if json fails
            except:
                return []
    return []

# Parse the stats column
fixtures['stats'] = fixtures['stats'].apply(parse_stats)

# Create the fixtures_stat_table
stats_records = []

for _, row in fixtures.iterrows():
    match_id = row['match_id']
    gw_id = row['gw_id']
    stats_list = row['stats']
    
    # Dictionary to hold all stats for this match
    match_stats = {
        'match_id': match_id,
        'gw_id': gw_id,
        'home_team_id': row['home_team_id'],
        'away_team_id': row['away_team_id'],
        'home_score': row['home_score'],
        'away_score': row['away_score'],
        'team_h_difficulty': row['team_h_difficulty'],
        'team_a_difficulty': row['team_a_difficulty']
    }
    
    # Loop through each stat type
    for stat in stats_list:
        identifier = stat['identifier']
        
        # Sum up home team stats (team_h)
        home_total = sum([s.get('value', 0) for s in stat.get('h', [])])
        match_stats[f'home_{identifier}'] = home_total
        
        # Sum up away team stats (team_a)
        away_total = sum([s.get('value', 0) for s in stat.get('a', [])])
        match_stats[f'away_{identifier}'] = away_total
    
    stats_records.append(match_stats)

# Create the new table
fixtures_stat_table = pd.DataFrame(stats_records)

# %%
"""
# Add match_id, if was_home is true then "team" comes first, if else then "opp_team_name" comes first
fpl20_24["match_id"] = (
    fpl20_24["season"] + "-" + 
    np.where(fpl20_24["was_home"], fpl20_24["team"], fpl20_24["opp_team_name"]) + "-" +
    np.where(fpl20_24["was_home"], fpl20_24["opp_team_name"], fpl20_24["team"])
)

"""

# %%
fpl20_24 = fpl20_24.drop(columns= ['position','team','xP','kickoff_time','fixture','transfers_balance','was_home','team_a_score','team_h_score','season','opp_team_name','home_team_id',"away_team_id"])
fpl20_24 = fpl20_24.rename(columns= {'selected':'selected_by_percent', 'value' : "now_cost"})

# %%
players_df.to_csv("seed_data/players.csv", index=False)
teams_df.to_csv("seed_data/teams.csv", index=False)
positions_df.to_csv("seed_data/positions.csv", index=False)
fixtures_df.to_csv("seed_data/fixtures.csv", index=False)
fixtures_stat_table.to_csv("seed_data/fixtures_stats.csv", index=False)
fpl20_24.to_csv("seed_data/fpl20_24.csv", index= False)

# %% [markdown]
#

# %% [markdown]
#

# %% [markdown]
# ### Structure for Fantasy Premier League data model
# Fact tables: 
# * playergameweekstats - Contains data on every player and every gameweek from 2020 - t.d
#     - Player_id links to player dim table
# * fixturegameweekstats - Contains data on every fixture for every gameweek from 2020 - t.d
#   - Match_id links to fixture dim table
#
# Dimension tables: 
# * Players table
# * Team table
# * Player position table
# * Fixture table : Table with all the fixtures every gameweek from 2020 - t.d
# * Fantasy Premier League gameweek points table - contains average score and highest score, used for benchmark, only 2025 season
