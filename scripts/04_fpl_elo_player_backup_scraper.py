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
# # FPL ELO Player Stats — Combined Old Pipeline + FBref
#
# Combines two data sources to produce per-match player statistics:
#
# **Old pipeline (FPL-Core-Insights / Opta)** — primary source for all stat columns.
# - `FPL-Core-Insights/data/2025-2026/By Tournament/Premier League/GW{n}/playermatchstats.csv`
# - Rich Opta stats: touches, accurate_passes, xgot, dribbles, duels, etc.
# - Degrades after ~GW26 for some columns; GW33+ is empty.
#
# **FBref (via `soccerdata`)** — structural base + gap-filler.
# - Provides match/player identity and position for all GWs.
# - Fills nulls left by old pipeline degradation.
# - Sole source for GW27+ rows where old pipeline is empty.
# - `start_min` / `finish_min` computed from FBref minutes-played (old pipeline values unreliable).
#
# **Output:** `FPL_DATA/elo_gameweek_fact.csv` (40 columns)

# %%
# # !pip install soccerdata rapidfuzz requests beautifulsoup4
import soccerdata as sd
import pandas as pd
import numpy as np
import re
import requests
import time
import sys
import os
from pathlib import Path
from bs4 import BeautifulSoup, Comment
from rapidfuzz import process, fuzz
import warnings
warnings.filterwarnings("ignore")

# Old version schema — 42 columns (unchanged)
ELO_PLAYER_COLS = [
    "match_id", "player_id", "team_id", "position_id", "gw_id", "was_home",
    "total_shots", "shots_on_target", "successful_dribbles", "big_chances_missed",
    "touches_opposition_box", "touches", "accurate_passes", "chances_created",
    "final_third_passes", "accurate_crosses", "accurate_long_balls", "interceptions",
    "recoveries", "blocks", "clearances", "headed_clearances", "dribbled_past",
    "duels_won", "duels_lost", "ground_duels_won", "aerial_duels_won",
    "was_fouled", "fouls_committed", "saves", "xgot_faced", "goals_prevented",
    "sweeper_actions", "gk_accurate_passes", "gk_accurate_long_balls", "high_claim",
    "offsides", "xgot", "start_min", "finish_min", "team_goals_conceded",
    "penalties_scored",
]

TEAM_NORM = {
    "Manchester Utd":    "Man Utd",
    "Manchester City":   "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Spurs",
    "Newcastle United":  "Newcastle",
    "West Ham United":   "West Ham",
    "Ipswich Town":      "Ipswich",
    "Leicester City":    "Leicester",
    "Leeds United":      "Leeds",
}

# FBref position codes → abbreviation (matches position_dim.csv)
POS_MAP = {
    "GK": "GK",
    "DF": "DEF", "CB": "DEF", "LB": "DEF", "RB": "DEF",
    "FB": "DEF", "WB": "DEF",
    "MF": "MID", "DM": "MID", "CM": "MID",
    "AM": "MID", "LM": "MID", "RM": "MID",
    "WM": "MID", "LW": "MID", "RW": "MID",
    "FW": "FWD", "SS": "FWD",
}

# Old pipeline full names → abbreviation (FPL-Core-Insights uses full names)
POS_FULL_TO_ABBR = {
    "Goalkeeper": "GK",
    "Defender":   "DEF",
    "Midfielder": "MID",
    "Forward":    "FWD",
}

GK_COLS = ["saves"]

SUMMARY_MAP = {
    "Performance_Sh":   "total_shots",
    "Performance_SoT":  "shots_on_target",
    "Performance_Fls":  "fouls_committed",
    "Performance_Fld":  "was_fouled",
    "Performance_Off":  "offsides",
    "Performance_PK":   "penalties_scored",
    "Performance_Int":  "interceptions",
}

PASSING_MAP    = {}
DEFENSE_MAP    = {}
POSSESSION_MAP = {}
MISC_MAP       = {}
KEEPER_MAP     = {"Shot Stopping_Saves": "saves"}

# FBref overrides these old-pipeline columns from GW26 onwards (where Opta data degrades)
FBREF_OVERRIDE_COLS = [
    "total_shots", "shots_on_target", "interceptions",
    "was_fouled", "fouls_committed", "offsides", "penalties_scored",
]
FBREF_OVERRIDE_GW = 26

print(f"Setup complete. Target columns: {len(ELO_PLAYER_COLS)}")

# %%
player_dim   = pd.read_csv("FPL_DATA/player_dim.csv")
position_dim = pd.read_csv("FPL_DATA/position_dim.csv")
team_dim     = pd.read_csv("FPL_DATA/team_dim.csv")

team_name_to_id = dict(zip(team_dim["team"], team_dim["team_id"]))
pos_name_to_id  = dict(zip(position_dim["position"], position_dim["position_id"]))

print(f"player_dim:   {len(player_dim)} rows")
print(f"position_dim: {pos_name_to_id}")
print(f"team_dim:     {len(team_dim)} rows")

# %%
fbref = sd.FBref(leagues="ENG-Premier League", seasons="2526")
sched = fbref.read_schedule()
print("Schedule shape:", sched.shape)
print("Index names:   ", sched.index.names)
print("Columns:       ", sched.columns.tolist())
sched.head(3)


# %%
def _find_col(df, candidates):
    """Return the first matching column name from candidates list, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: partial match
    for key in candidates:
        matches = [col for col in df.columns if key.lower() in col.lower()]
        if matches:
            return matches[0]
    return None


sched_flat = sched.reset_index()

# Identify round/week column (name varies by soccerdata version)
round_col = _find_col(sched_flat, ["round", "week", "matchweek"])
assert round_col, f"No round column found in: {sched_flat.columns.tolist()}"
print(f"Using: round={round_col!r}")

# soccerdata returns a combined 'score' column (e.g. '4–2') — parse into home/away goals
score_parts = sched_flat["score"].str.extract(r"(\d+)\D+(\d+)")
sched_flat["home_goals"] = pd.to_numeric(score_parts[0], errors="coerce")
sched_flat["away_goals"] = pd.to_numeric(score_parts[1], errors="coerce")

sched_flat["gw"] = pd.to_numeric(sched_flat[round_col], errors="coerce").astype("Int64")

sched_flat["home_team_norm"] = sched_flat["home_team"].map(TEAM_NORM).fillna(sched_flat["home_team"])
sched_flat["away_team_norm"] = sched_flat["away_team"].map(TEAM_NORM).fillna(sched_flat["away_team"])
sched_flat["home_team_id"]   = sched_flat["home_team_norm"].map(team_name_to_id)
sched_flat["away_team_id"]   = sched_flat["away_team_norm"].map(team_name_to_id)

unmapped = sched_flat[
    sched_flat["home_team_id"].isna() | sched_flat["away_team_id"].isna()
]["home_team_norm"].dropna().unique()
if len(unmapped):
    print(f"WARNING — unmapped teams (add to TEAM_NORM): {list(unmapped)}")

sched_flat = sched_flat.dropna(subset=["home_team_id", "away_team_id", "gw"])
sched_flat = sched_flat.astype({"home_team_id": int, "away_team_id": int, "gw": int})

sched_flat["match_id"] = (
    "2025"
    + sched_flat["home_team_id"].astype(str).str.zfill(2)
    + sched_flat["away_team_id"].astype(str).str.zfill(2)
).astype(int)
sched_flat["gw_id"] = (
    sched_flat["match_id"].astype(str).str[:4].astype(int) * 100 + sched_flat["gw"]
)

game_meta = sched_flat.set_index("game_id")[
    ["match_id", "gw_id", "home_team_id", "away_team_id", "home_goals", "away_goals"]
].copy()

print(f"game_meta: {len(game_meta)} fixtures, "
      f"{game_meta['match_id'].notna().sum()} with match_id")
game_meta.head(3)

# %%
import unicodedata

def normalize_name(name):
    """Strip accents for fuzzy matching."""
    if not isinstance(name, str):
        return name
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("ascii")

name_to_id      = dict(zip(player_dim["full_name"], player_dim["player_id"]))
name_to_id_norm = {normalize_name(k): v for k, v in name_to_id.items()}
all_names_norm  = list(name_to_id_norm.keys())

# Secondary lookup by web_name for single-name players (e.g. "Beto", "Casemiro", "Rodri")
webname_to_id = dict(zip(player_dim["web_name"], player_dim["player_id"]))


def resolve_player_id(name, threshold=90):
    if pd.isna(name) or not str(name).strip():
        return None
    norm = normalize_name(str(name))
    # 1. Exact match on normalised full name
    if norm in name_to_id_norm:
        return int(name_to_id_norm[norm])
    # 2. Exact match on web_name (catches single-name players like "Beto", "Casemiro")
    if name in webname_to_id:
        return int(webname_to_id[name])
    # 3. Fuzzy match on normalised full name
    result = process.extractOne(norm, all_names_norm, scorer=fuzz.token_set_ratio)
    return int(name_to_id_norm[result[0]]) if result and result[1] >= threshold else None


print(f"Player name cache: {len(name_to_id)} entries")

# %%
from lxml import html as lhtml

PAGE_CACHE = Path.home() / "soccerdata" / "data" / "FBref" / "match_pages"
PAGE_CACHE.mkdir(parents=True, exist_ok=True)

# Matches both stats_X_summary/passing/etc AND keeper_stats_X
_STAT_RE = re.compile(
    r"^(stats_\w+_(summary|passing|defense|possession|misc)|keeper_stats_\w+)$"
)

def fetch_full_page(game_id):
    cache_file = PAGE_CACHE / f"{game_id}.html"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    driver = fbref._driver
    driver.get(f"https://fbref.com/en/matches/{game_id}")
    time.sleep(4)

    page_html = driver.page_source
    tree  = lhtml.fromstring(page_html)
    bodies = tree.xpath("//body")
    if not bodies:
        raise RuntimeError(f"No <body> for {game_id}")
    body_html = lhtml.tostring(bodies[0], encoding="unicode")
    full_html = f"<html><head><meta charset='utf-8'></head>{body_html}</html>"
    cache_file.write_text(full_html, encoding="utf-8")
    return full_html


def flatten_df_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(
                str(x) for x in col
                if str(x) and not str(x).lower().startswith("unnamed")
            ).strip("_")
            for col in df.columns
        ]
    return df


def parse_page(html_str, game_id):
    soup = BeautifulSoup(html_str, "html.parser")
    for c in soup.find_all(
        string=lambda t: isinstance(t, Comment) and "<table" in t
    ):
        c.replace_with(BeautifulSoup(c, "html.parser"))

    tables = []
    for tbl in soup.find_all("table", id=_STAT_RE):
        tbl_id = tbl["id"]
        stat_type = "keeper" if tbl_id.startswith("keeper_stats_") else tbl_id.rsplit("_", 1)[1]

        caption = tbl.find("caption")
        team_raw = ""
        if caption:
            text = caption.get_text(strip=True)
            m = re.match(r"^(.*?)\s+(?:Player|Goalkeeper)\s+Stats", text)
            team_raw = m.group(1) if m else text

        try:
            df = pd.read_html(str(tbl), header=[0, 1])[0]
        except Exception:
            continue
        df = flatten_df_cols(df)
        pcol = next(
            (c for c in df.columns if c.lower().rstrip("_") == "player"), None
        )
        if pcol is None:
            continue
        df = df.rename(columns={pcol: "_player"})
        df = df[
            df["_player"].notna()
            & ~df["_player"].astype(str).str.contains("Squad|Total", na=False)
        ].copy()
        df["_stat_type"] = stat_type
        df["_team_raw"]  = team_raw
        df["game_id"]    = game_id
        tables.append(df)
    return tables


cached = len(list(PAGE_CACHE.glob("*.html")))
print(f"Cache dir : {PAGE_CACHE}")
print(f"Cached    : {cached} / {len(sched_flat)} pages")
print(f"Driver    : {'ready' if hasattr(fbref, '_driver') else 'NOT FOUND'}")

# %%
# Uses fbref._driver (soccerdata Chrome) -- bypasses Cloudflare, no rate limit needed.
# First run ~30 min (330 pages x ~6 s each). Subsequent runs instant from cache.
all_tables = []
rows   = sched_flat[sched_flat["match_report"].notna()]
total  = len(rows)
errors = []
CR = chr(13)

for i, row in enumerate(rows.itertuples(), 1):
    msg = f"[{i:3d}/{total}] {row.game_id}  {row.home_team_norm} vs {row.away_team_norm}"
    print(msg, end=CR, flush=True)
    try:
        html_str = fetch_full_page(row.game_id)
        tables   = parse_page(html_str, row.game_id)
        all_tables.extend(tables)
    except Exception as e:
        errors.append((row.game_id, str(e)))
        print(f"ERROR {row.game_id}: {e}")

stat_counts = {}
for t in all_tables:
    st = t["_stat_type"].iloc[0]
    stat_counts[st] = stat_counts.get(st, 0) + 1

print(f"Done. {len(all_tables)} table DataFrames from {total} matches.")
print("Tables per stat type:", stat_counts)
if errors:
    print(f"Errors ({len(errors)}): {errors[:5]}")


# %%
def get_type(stat_type):
    dfs = [t for t in all_tables if t["_stat_type"].iloc[0] == stat_type]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

summary_df    = get_type("summary")
passing_df    = get_type("passing")
defense_df    = get_type("defense")
possession_df = get_type("possession")
misc_df       = get_type("misc")
keeper_df     = get_type("keeper")

print("Rows per stat type:")
for name, df in [
    ("summary",    summary_df),
    ("passing",    passing_df),
    ("defense",    defense_df),
    ("possession", possession_df),
    ("misc",       misc_df),
    ("keeper",     keeper_df),
]:
    print(f"  {name:12s}: {len(df):6,}")

print("\nSummary columns:")
for c in summary_df.columns.tolist():
    print(f"  {c}")
print("\nPassing columns:")
for c in passing_df.columns.tolist():
    print(f"  {c}")
print("\nDefense columns:")
for c in defense_df.columns.tolist():
    print(f"  {c}")
print("\nPossession columns:")
for c in possession_df.columns.tolist():
    print(f"  {c}")
print("\nMisc columns:")
for c in misc_df.columns.tolist():
    print(f"  {c}")
print("\nKeeper columns:")
for c in keeper_df.columns.tolist():
    print(f"  {c}")

# %%
# ── OLD PIPELINE: Full processing (mirrors fpl_elo_old_version.ipynb) ─────────
# Primary data source for all GWs where FPL-Core-Insights has data.
# team_goals_conceded, start_min, finish_min, xgot etc. come directly from here.

ELO_BASE = "FPL-Core-Insights/data/2025-2026"
ELO_PL   = f"{ELO_BASE}/By Tournament/Premier League"

elo_players_raw = pd.read_csv(f"{ELO_BASE}/players.csv")
elo_teams_raw   = pd.read_csv(f"{ELO_BASE}/teams.csv")
elo_players_raw["full_name"] = elo_players_raw["first_name"] + " " + elo_players_raw["second_name"]

# Load all GW playermatchstats, joining fixtures for home/away team codes
old_frames = []
for gw in range(1, 39):
    fpath  = os.path.join(ELO_PL, f"GW{gw}", "playermatchstats.csv")
    fxpath = os.path.join(ELO_PL, f"GW{gw}", "fixtures.csv")
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    if df.empty or len(df.columns) < 5:
        continue
    df["GW"] = gw
    if os.path.exists(fxpath):
        fx = pd.read_csv(fxpath)[["match_id", "home_team", "away_team"]]
        df = df.merge(fx, on="match_id", how="left")
    df = df.dropna(axis=1, how="all")
    old_frames.append(df)

old_raw = pd.concat(old_frames, ignore_index=True)

# Drop columns already covered elsewhere or not needed
_drop = ["minutes_played","goals","assists","xg","xa","penalties_missed",
         "tackles","goals_conceded"]
old_raw = old_raw.drop(columns=[c for c in _drop if c in old_raw.columns])

# Translate FPL-Core player_id → full_name, then get PBI player_id via full_name
old_raw = pd.merge(old_raw, elo_players_raw[["player_id","full_name"]],
                   how="left", on="player_id").drop(columns=["player_id"])

# Map home/away team codes → names → PBI team IDs
old_raw = (pd.merge(old_raw, elo_teams_raw[["code","name"]], how="left",
                    left_on="home_team", right_on="code")
           .drop(columns=["code"]).rename(columns={"name":"home_team_name"}))
old_raw = (pd.merge(old_raw, elo_teams_raw[["code","name"]], how="left",
                    left_on="away_team", right_on="code")
           .drop(columns=["code"]).rename(columns={"name":"away_team_name"}))
old_raw = (pd.merge(old_raw, team_dim, how="left",
                    left_on="home_team_name", right_on="team")
           .drop(columns=["team"]).rename(columns={"team_id":"home_team_id"}))
old_raw = (pd.merge(old_raw, team_dim, how="left",
                    left_on="away_team_name", right_on="team")
           .drop(columns=["team"]).rename(columns={"team_id":"away_team_id"}))

# Build match_id and gw_id (same formula as FBref pipeline)
old_raw["match_id"] = (
    "2025"
    + old_raw["home_team_id"].astype(str).str.zfill(2)
    + old_raw["away_team_id"].astype(str).str.zfill(2)
).astype("Int64")
old_raw["gw_id"] = (
    old_raw["match_id"].astype(str).str[:4].astype(int) * 100
    + old_raw["GW"].astype(int)
)

# Resolve PBI player_id via full_name
old_raw = pd.merge(old_raw, player_dim[["player_id","full_name"]],
                   how="left", on="full_name")

# Resolve team_id: player's team code → team name → PBI team_id
old_raw = pd.merge(old_raw, elo_players_raw[["full_name","team_code","position"]],
                   how="left", on="full_name")
old_raw = (pd.merge(old_raw, elo_teams_raw[["code","name"]], how="left",
                    left_on="team_code", right_on="code")
           .drop(columns=["team_code","code"]))
old_raw = (pd.merge(old_raw, team_dim, how="left", left_on="name", right_on="team")
           .drop(columns=["name","team"]))

# Resolve position_id (old pipeline uses full names; map to abbreviations first)
old_raw["position_id"] = old_raw["position"].map(POS_FULL_TO_ABBR).map(pos_name_to_id)
old_raw = old_raw.drop(columns=["position"], errors="ignore")

# was_home
old_raw["was_home"] = (old_raw["team_id"] == old_raw["home_team_id"]).astype(int)

# Ensure every target column exists (fill missing with 0, matching old version convention)
for col in ELO_PLAYER_COLS:
    if col not in old_raw.columns:
        old_raw[col] = 0

old_df = old_raw[ELO_PLAYER_COLS].copy()

gws = sorted(old_raw["GW"].dropna().unique().astype(int))
print(f"Old pipeline: {len(old_df):,} rows  GW{min(gws)}–GW{max(gws)}")
print(f"Null player_id: {old_df['player_id'].isna().sum()} rows")
print(f"Null position_id: {old_df['position_id'].isna().sum()} rows")

# %%
# ── STEP 1: Build FBref summary lookup (match_id + player_id keys) ───────────
_fb = summary_df.copy()
for src, dst in [(["pos","position"],"_pos"),(["min","mins"],"_min_raw")]:
    col = next((c for c in _fb.columns if c.lower().rstrip("_") in src), None)
    if col and col != dst:
        _fb = _fb.rename(columns={col: dst})

_fb = _fb.rename(columns={k: v for k, v in SUMMARY_MAP.items() if k in _fb.columns})

# Join keeper saves
_kslim = (keeper_df[["game_id","_player","Shot Stopping_Saves"]]
          .rename(columns={"Shot Stopping_Saves":"saves"})
          .drop_duplicates(["game_id","_player"]))
_fb = _fb.merge(_kslim, on=["game_id","_player"], how="left")

# Join game metadata
_fb = _fb.merge(game_meta.reset_index(), on="game_id", how="left")

# Resolve player_id (unicode-normalised fuzzy match)
_fb["player_id"] = _fb["_player"].apply(resolve_player_id)
_fb["player_id"] = pd.to_numeric(_fb["player_id"], errors="coerce").astype("Int64")

# Resolve team
_fb["team_norm"]  = _fb["_team_raw"].map(TEAM_NORM).fillna(_fb["_team_raw"])
_fb["team_id"]    = _fb["team_norm"].map(team_name_to_id).astype("Int64")

# Resolve position
_fb["_pos_primary"] = _fb["_pos"].str.split(",").str[0].str.strip()
_fb["position_id"]  = _fb["_pos_primary"].map(POS_MAP).map(pos_name_to_id).astype("Int64")

# Derived columns needed for FBref-only rows (GW34+)
_min = pd.to_numeric(_fb.get("_min_raw", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
_fb["was_home"]            = (_fb["team_id"] == _fb["home_team_id"]).astype("Int64")
_fb["team_goals_conceded"] = np.where(_fb["was_home"]==1, _fb["away_goals"], _fb["home_goals"])
_is_starter = _min >= 60
_fb["start_min"]  = np.where(_is_starter, 0, (90 - _min).clip(lower=0))
_fb["finish_min"] = np.where(_is_starter, _min, 90)

# Slim version for override merge (only override cols)
fbref_override = (
    _fb[["match_id","player_id"] + FBREF_OVERRIDE_COLS]
    .drop_duplicates(["match_id","player_id"])
)

# ── STEP 2: Override degraded columns for GW26+ in old pipeline data ─────────
gw_num = old_df["gw_id"].astype(int) % 100

old_early = old_df[gw_num < FBREF_OVERRIDE_GW].copy()
old_late  = old_df[gw_num >= FBREF_OVERRIDE_GW].copy()

old_late = old_late.merge(
    fbref_override.rename(columns={c: f"_fb_{c}" for c in FBREF_OVERRIDE_COLS}),
    on=["match_id","player_id"], how="left",
)
for col in FBREF_OVERRIDE_COLS:
    fb_col = f"_fb_{col}"
    if fb_col in old_late.columns:
        # FBref wins where it has a value; old pipeline kept otherwise
        old_late[col] = old_late[fb_col].fillna(old_late[col])
        old_late.drop(columns=[fb_col], inplace=True, errors="ignore")

df_combined = pd.concat([old_early, old_late], ignore_index=True)

# ── STEP 3: Add FBref rows for any (match_id, player_id) not in old pipeline ─
# Uses (match_id, player_id) pairs so players missing from old pipeline for
# specific GWs (e.g. GW30-33 gaps) are filled in from FBref.
old_keys = pd.MultiIndex.from_arrays([
    df_combined["match_id"].dropna().astype(int),
    df_combined["player_id"].dropna().astype(int),
])
_fb_valid = _fb[_fb["match_id"].notna() & _fb["player_id"].notna()].copy()
fb_keys = pd.MultiIndex.from_arrays([
    _fb_valid["match_id"].astype(int),
    _fb_valid["player_id"].astype(int),
])
fb_extra = _fb_valid[~fb_keys.isin(old_keys)].copy()

if not fb_extra.empty:
    for col in ELO_PLAYER_COLS:
        if col not in fb_extra.columns:
            fb_extra[col] = np.nan
    df_out = pd.concat([df_combined, fb_extra[ELO_PLAYER_COLS]], ignore_index=True)
    print(f"Added {len(fb_extra)} FBref supplement rows")
else:
    df_out = df_combined.copy()

df_out = df_out[ELO_PLAYER_COLS].copy()
print(f"Output shape: {df_out.shape}")

gw_counts = df_out.groupby(df_out["gw_id"].astype(int) % 100).size()
print("\nPlayer rows per GW:")
print(gw_counts.to_string())
df_out.head(3)

# %%
# Null / zero rates for key stat columns
print("=== Data quality check ===\n")

stat_cols = [c for c in ELO_PLAYER_COLS if c not in
             ["match_id","player_id","team_id","position_id","gw_id","was_home"]]

null_rate = df_out[stat_cols].isna().mean().round(3)
zero_rate = (df_out[stat_cols] == 0).mean().round(3)

quality = pd.DataFrame({"null_%": null_rate, "all_zero_%": zero_rate})
has_issues = quality[(quality["null_%"] > 0) | (quality["all_zero_%"] > 0.95)]
if not has_issues.empty:
    print("Columns with nulls or >95% zeros:")
    print(has_issues.to_string())
else:
    print("All stat columns look clean")

print("\nID column nulls:")
for col in ["match_id","player_id","team_id","position_id"]:
    n = df_out[col].isna().sum()
    print(f"  {col}: {n} nulls ({n/len(df_out):.1%})")

# %%
df_save = df_out[df_out["player_id"].notna()].copy()

# Ensure total_shots >= goals_scored (prevents >100% conversion rate when GW rows are sparse)
fpl_gw = pd.read_csv("FPL_DATA/fpl_gameweek_fact.csv", usecols=["match_id", "player_id", "goals_scored"])
fpl_gw = fpl_gw[fpl_gw["goals_scored"] > 0].copy()
df_save = df_save.merge(fpl_gw, on=["match_id", "player_id"], how="left")
df_save["total_shots"] = df_save[["total_shots", "goals_scored"]].max(axis=1)
df_save = df_save.drop(columns=["goals_scored"])

df_save.to_csv("FPL_DATA/elo_gameweek_fact.csv", index=False)
print(f"Saved: FPL_DATA/elo_gameweek_fact.csv  ({len(df_save)} rows, {len(df_save.columns)} cols)")
print(f"Dropped {len(df_out) - len(df_save)} rows with null player_id")

# %%
# ── DEBUG: Show unmatched FBref player names ──────────────────────────────────
unmatched_fb = _fb[_fb["player_id"].isna()][["_player", "_team_raw", "match_id", "gw_id"]].drop_duplicates("_player")

print(f"Unmatched FBref players: {len(unmatched_fb)}\n")
results = []
for _, row in unmatched_fb.iterrows():
    name = row["_player"]
    norm = normalize_name(str(name))
    best = process.extractOne(norm, all_names_norm, scorer=fuzz.token_set_ratio)
    results.append({
        "fbref_name":  name,
        "team":        row["_team_raw"],
        "best_match":  best[0] if best else None,
        "score":       best[1] if best else 0,
    })

debug_df = pd.DataFrame(results).sort_values("score", ascending=False)
print(debug_df.to_string(index=False))
