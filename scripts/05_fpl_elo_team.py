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

# %% [markdown]
# ## FPL ELO Fixtures Stats — Premier League Official API
#
# Fetches match-level team statistics from the **Premier League's official stats API**  
# (`footballapi.pulselive.com`) as a replacement for the FPL-Core-Insights GitHub repo.
#
# **Pipeline:**
# 1. Fetch all fixtures with `pulse_id` from the FPL API (GW1–38)
# 2. For each fixture, call `/football/stats/match/{pulse_id}` on the PL API
# 3. Extract home/away stats, map to target schema
# 4. Compute `match_id` and `gw_id` using the same formula as `fpl_elo.ipynb`
#
# **Coverage vs elo_fixture_fact.csv:**
# - ✓ ~30/38 stat pairs directly available  
# - ✗ xG variants (`expected_goals_xg`, `xg_open_play`, etc.) — not in PL API  
# - ✗ `own_half` / `opposition_half` touches — not in PL API  
# - ✗ `home/away_team_elo` — calculated, not scraped  
#
# **Output:** `FPL_DATA/elo_fixture_fact.csv`

# %%
import requests
import pandas as pd
import numpy as np
import time
import json
import os

# %%
# Load FPL lookup tables (same as fpl_elo.ipynb)
team_dim = pd.read_csv('FPL_DATA/team_dim.csv')

# Exact column schema from elo_fixture_fact.csv
ELO_FIXTURES_COLS = [
    'match_id', 'gw_id', 'home_team_id', 'away_team_id', 'home_team_elo', 'away_team_elo',
    'home_possession', 'away_possession', 'home_expected_goals_xg', 'away_expected_goals_xg',
    'home_total_shots', 'away_total_shots', 'home_shots_on_target', 'away_shots_on_target',
    'home_big_chances', 'away_big_chances', 'home_big_chances_missed', 'away_big_chances_missed',
    'home_accurate_passes', 'away_accurate_passes', 'home_fouls_committed', 'away_fouls_committed',
    'home_corners', 'away_corners', 'home_xg_open_play', 'away_xg_open_play',
    'home_xg_set_play', 'away_xg_set_play', 'home_non_penalty_xg', 'away_non_penalty_xg',
    'home_xg_on_target_xgot', 'away_xg_on_target_xgot',
    'home_shots_off_target', 'away_shots_off_target', 'home_blocked_shots', 'away_blocked_shots',
    'home_hit_woodwork', 'away_hit_woodwork', 'home_shots_inside_box', 'away_shots_inside_box',
    'home_shots_outside_box', 'away_shots_outside_box', 'home_passes', 'away_passes',
    'home_own_half', 'away_own_half', 'home_opposition_half', 'away_opposition_half',
    'home_accurate_long_balls', 'away_accurate_long_balls',
    'home_accurate_crosses', 'away_accurate_crosses', 'home_throws', 'away_throws',
    'home_touches_in_opposition_box', 'away_touches_in_opposition_box',
    'home_offsides', 'away_offsides', 'home_yellow_cards', 'away_yellow_cards',
    'home_red_cards', 'away_red_cards', 'home_tackles_won', 'away_tackles_won',
    'home_interceptions', 'away_interceptions', 'home_blocks', 'away_blocks',
    'home_clearances', 'away_clearances', 'home_keeper_saves', 'away_keeper_saves',
    'home_duels_won', 'away_duels_won', 'home_ground_duels_won', 'away_ground_duels_won',
    'home_aerial_duels_won', 'away_aerial_duels_won',
    'home_successful_dribbles', 'away_successful_dribbles',
]

# PL API stat name -> target column name (without home_/away_ prefix)
# Notes:
#   possession_percentage: PL API returns one decimal; we round to integer to match elo source
#   total_tackle: elo source counts total tackle attempts (not just won) despite the column name
PL_STAT_MAP = {
    'possession_percentage': 'possession',
    'total_scoring_att':     'total_shots',
    'ontarget_scoring_att':  'shots_on_target',
    'shot_off_target':       'shots_off_target',
    'blocked_scoring_att':   'blocked_shots',
    'hit_woodwork':          'hit_woodwork',
    'attempts_ibox':         'shots_inside_box',
    'attempts_obox':         'shots_outside_box',
    'big_chance_created':    'big_chances',
    'big_chance_missed':     'big_chances_missed',
    'total_pass':            'passes',
    'accurate_pass':         'accurate_passes',
    'accurate_long_balls':   'accurate_long_balls',
    'accurate_cross':        'accurate_crosses',
    'won_corners':           'corners',
    'total_throws':          'throws',
    'touches_in_opp_box':    'touches_in_opposition_box',
    'total_offside':         'offsides',
    'total_yel_card':        'yellow_cards',
    'total_red_card':        'red_cards',
    'total_tackle':          'tackles_won',   # elo source = total attempts, not just won
    'interception_won':      'interceptions',
    'outfielder_block':      'blocks',
    'effective_clearance':   'clearances',
    'saves':                 'keeper_saves',
    'duel_won':              'duels_won',
    'aerial_won':            'aerial_duels_won',
    'won_contest':           'successful_dribbles',
    'fk_foul_lost':          'fouls_committed',
}

print('Setup complete.')
print(f'Target columns: {len(ELO_FIXTURES_COLS)}')
print(f'PL API stat mappings: {len(PL_STAT_MAP)}')

# %% [markdown]
# ## Step 1 — Get Fixtures & pulse_id from FPL API

# %%
FPL_HEADERS = {'User-Agent': 'Mozilla/5.0'}
PL_HEADERS  = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Origin':  'https://www.premierleague.com',
    'Referer': 'https://www.premierleague.com/',
}

# Build FPL API team_id → persistent team_id (from team_dim) using team name as key
# FPL reassigns team IDs each season as promoted/relegated teams change; team_dim uses
# persistent cross-season IDs. We derive the mapping dynamically so it works every season.
_bootstrap = requests.get(
    'https://fantasy.premierleague.com/api/bootstrap-static/',
    headers=FPL_HEADERS, timeout=15
).json()
_api_teams = pd.DataFrame(_bootstrap['teams'])[['id', 'name']].rename(columns={'id': 'fpl_api_id', 'name': 'team'})
_fpl_to_persistent = (
    _api_teams.merge(team_dim, on='team', how='left')
    .set_index('fpl_api_id')['team_id']
    .to_dict()
)
print('FPL API id -> persistent team_id mapping:')
for fpl_id, pid in sorted(_fpl_to_persistent.items()):
    name = _api_teams.set_index('fpl_api_id').loc[fpl_id, 'team']
    print(f'  {fpl_id:3d} ({name:<18}) -> {pid}')


def fetch_fpl_fixtures(gw_range=range(1, 39)):
    """Fetch all FPL fixtures for given GWs. Returns DataFrame with pulse_id and persistent team IDs."""
    rows = []
    for gw in gw_range:
        r = requests.get(
            f'https://fantasy.premierleague.com/api/fixtures/?event={gw}',
            headers=FPL_HEADERS, timeout=10
        )
        if r.status_code != 200:
            print(f'  GW{gw}: HTTP {r.status_code}')
            continue
        for fix in r.json():
            if not fix.get('finished', False):
                continue
            # Map FPL season-specific IDs -> persistent team_dim IDs
            home_id = _fpl_to_persistent.get(fix['team_h'])
            away_id = _fpl_to_persistent.get(fix['team_a'])
            rows.append({
                'gw':          gw,
                'pulse_id':    fix['pulse_id'],
                'team_h':      home_id,
                'team_a':      away_id,
                'home_score':  fix['team_h_score'],
                'away_score':  fix['team_a_score'],
            })
    df = pd.DataFrame(rows)
    df['season']   = '2025-26'
    df['match_id'] = (
        df['season'].str[:4]
        + df['team_h'].astype(str).str.zfill(2)
        + df['team_a'].astype(str).str.zfill(2)
    ).astype('Int64')
    df['gw_id'] = df['match_id'].astype(str).str[:4].astype(int) * 100 + df['gw']
    return df


print()
print('Fetching FPL fixtures GW1-38...')
fixtures = fetch_fpl_fixtures()
print(f'Finished matches: {len(fixtures)}')
print(f'GW range: {fixtures["gw"].min()}-{fixtures["gw"].max()}')
fixtures.head(3)

# %% [markdown]
# ## Step 2 — Fetch PL API Match Stats

# %%
# Optional: cache raw JSON responses so re-running is instant
CACHE_DIR = 'pl_api_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_pl_match_stats(pulse_id: int) -> dict:
    """Fetch match stats from PL API, using local JSON cache."""
    cache_path = os.path.join(CACHE_DIR, f'{pulse_id}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    r = requests.get(
        f'https://footballapi.pulselive.com/football/stats/match/{pulse_id}',
        headers=PL_HEADERS, timeout=15
    )
    if r.status_code != 200:
        return {}
    data = r.json()
    with open(cache_path, 'w') as f:
        json.dump(data, f)
    return data


def extract_team_stats(stat_list: list) -> dict:
    """Convert PL API stats list [{'name':..,'value':..}] to dict."""
    return {s['name']: s['value'] for s in stat_list}


def build_fixture_row(fpl_row: dict, pl_data: dict) -> dict:
    """Build one wide row for elo_fixture_fact from FPL + PL API data."""
    row = {
        'match_id':     fpl_row['match_id'],
        'gw_id':        fpl_row['gw_id'],
        'home_team_id': fpl_row['team_h'],
        'away_team_id': fpl_row['team_a'],
    }

    entity = pl_data.get('entity', {})
    teams  = entity.get('teams', [])
    data   = pl_data.get('data', {})

    if len(teams) < 2 or not data:
        return row  # match not yet played or API returned no stats

    # teams[0] = home, teams[1] = away (PL API convention)
    home_pl_id = str(teams[0]['team']['id'])
    away_pl_id = str(teams[1]['team']['id'])

    home_stats = extract_team_stats(data.get(home_pl_id, {}).get('M', []))
    away_stats = extract_team_stats(data.get(away_pl_id, {}).get('M', []))

    # PL API omits stats with value 0 — default to 0 so nulls mean "no data", not "zero"
    for pl_name, target_suffix in PL_STAT_MAP.items():
        row[f'home_{target_suffix}'] = home_stats.get(pl_name, 0)
        row[f'away_{target_suffix}'] = away_stats.get(pl_name, 0)

    # Possession: PL API gives one decimal place; elo source stores as integer
    row['home_possession'] = round(row['home_possession'])
    row['away_possession'] = round(row['away_possession'])

    # Compute ground_duels_won = total duels won - aerial duels won
    for side in ('home', 'away'):
        row[f'{side}_ground_duels_won'] = row[f'{side}_duels_won'] - row[f'{side}_aerial_duels_won']

    return row


print('Helpers defined.')

# %%
rows = []
total = len(fixtures)

for i, fix in fixtures.iterrows():
    pulse_id = int(fix['pulse_id'])
    pl_data  = fetch_pl_match_stats(pulse_id)
    row      = build_fixture_row(fix.to_dict(), pl_data)
    rows.append(row)

    if (i + 1) % 50 == 0 or (i + 1) == total:
        print(f'  {i+1}/{total} done')

    # Polite rate limit — skip if already cached
    cache_path = f'{CACHE_DIR}/{pulse_id}.json'
    if not os.path.exists(cache_path):
        time.sleep(0.5)

print('All fixtures processed.')

# %%
pl_fixtures = pd.DataFrame(rows)

# Add all target columns not yet populated as None (xG, own_half, opposition_half, elo)
for col in ELO_FIXTURES_COLS:
    if col not in pl_fixtures.columns:
        pl_fixtures[col] = None

# Reorder to exact ELO schema
elo_fixture_fact = pl_fixtures[ELO_FIXTURES_COLS].copy()

print(f'Shape: {elo_fixture_fact.shape}')
elo_fixture_fact.head(3)

# %% [markdown]
# ## Step 3 — Coverage & Quality Check

# %%
# Rows per GW — verify coverage past GW26
gw_counts = elo_fixture_fact.groupby(elo_fixture_fact['gw_id'] % 100).size()
print('Fixtures per GW:')
print(gw_counts.to_string())

# %%
# Null rate per column
null_rate = elo_fixture_fact.isna().mean().round(2)

available = null_rate[null_rate == 0].index.tolist()
partial   = null_rate[(null_rate > 0) & (null_rate < 1)].index.tolist()
missing   = null_rate[null_rate == 1].index.tolist()

print(f'Fully populated ({len(available)}): {available}')
print(f'Partial ({len(partial)}): {partial}')
print(f'Not available ({len(missing)}): {missing}')

# %%
# Spot-check: compare a sample match against the existing elo_fixture_fact.csv before overwriting
elo_existing = pd.read_csv('FPL_DATA/elo_fixture_fact.csv')

common = sorted(set(elo_fixture_fact['match_id'].dropna()) & set(elo_existing['match_id'].dropna()))
print(f'Common match_ids: {len(common)}')

if common:
    sample_id = common[0]
    compare_cols = [
        'home_possession', 'away_possession',
        'home_total_shots', 'away_total_shots',
        'home_shots_on_target', 'away_shots_on_target',
        'home_big_chances', 'away_big_chances',
        'home_accurate_passes', 'away_accurate_passes',
        'home_corners', 'away_corners',
        'home_keeper_saves', 'away_keeper_saves',
    ]
    new_row = elo_fixture_fact[elo_fixture_fact['match_id'] == sample_id][compare_cols]
    old_row = elo_existing[elo_existing['match_id'] == sample_id][compare_cols]

    compare = pd.DataFrame({
        'PL API (new)': new_row.iloc[0],
        'ELO repo (old)': old_row.iloc[0],
    })
    print(f'\nSpot-check match_id={sample_id}:')
    print(compare.to_string())

# %% [markdown]
# ## Step 4 — Save

# %%
elo_fixture_fact.to_csv('FPL_DATA/elo_fixture_fact.csv', index=False)
print(f'Saved: FPL_DATA/elo_fixture_fact.csv  ({len(elo_fixture_fact)} rows, {len(elo_fixture_fact.columns)} cols)')

# %%
elo_fixture_fact
