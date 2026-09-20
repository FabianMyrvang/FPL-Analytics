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
# Lets this code find "FPL_DATA/" etc. whether it is run from notebooks/, scripts/, or the repo
# root. We chdir away from scripts/, so put it on sys.path explicitly to keep imports working.
import os
import sys
from pathlib import Path
_SCRIPTS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if Path.cwd().name in ("notebooks", "scripts"):
    os.chdir(Path.cwd().parent)

# %% [markdown]
# # Predict player points for the next three gameweeks
#
# Production script — runs in CI via `.github/workflows/predict-points.yml`.
#
# Models are refitted from scratch on every run rather than loaded from disk. Fitting takes well
# under a minute, and it removes model versioning, pickle compatibility across scikit-learn
# releases, and binary blobs in git history as entire classes of problem.
#
# See `06_fpl_model_backtest.py` for how these models were chosen and what they are worth.

# %%
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import upsert_csv, DIM_PLAYER, DIM_POSITION, DIM_TEAM
from model_common import (
    BLEND_MEMBERS, DIM_MODEL, EXPORT_DIR, FACT_PLAYER_POINTS_PREDICTION,
    FACT_PLAYER_PREDICTION_HISTORY, HORIZONS, MIN_TRAIN_SEASON, MODEL_DIM, PRIMARY_MODEL,
    XLSX_PREDICTIONS,
    add_baselines, add_player_features, build_fixture_spine, build_player_frame,
    feature_columns, fit_two_stage, predict_two_stage,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

KIND_TO_MODEL = {"linear": "linear_2stage", "gbm": "gbm_2stage",
                 "gbm_poisson": "gbm_poisson_2stage"}
RUN_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# Published per row as a DATE, not the full timestamp. The models are deterministic and the
# features only move when new results land, so on a quiet day every prediction is byte-identical
# to yesterday's — and a per-second timestamp would force a 1.5 MB commit anyway. With a date,
# the diff-gated commit step correctly does nothing. `data_through_gw_id` is the staleness signal
# that actually matters; the exact run time is kept in the Excel About sheet.
RUN_DATE = RUN_AT[:10]

# %%
spine = build_fixture_spine()
frame = build_player_frame(spine)

observed = frame[frame["played"].fillna(False) & frame["total_points"].notna()]
DATA_THROUGH = int(observed["gw_id"].max())
SEASON = int(frame["season"].max())
print(f"Data observed through gw_id {DATA_THROUGH}; predicting for season {SEASON}")

# %% [markdown]
# ## Which gameweeks are we predicting?
#
# Taken from the fixture list by kickoff time, never by adding 1 to `gw_id` — gameweeks can be
# blank for a given team, and the season prefix rolls over at the end of a season.

# %%
first_kickoff = (
    spine[spine["season"] == SEASON].groupby("gw_id")["kickoff_time"].min().sort_values()
)
# Only gameweeks that have not started yet. A gameweek already under way has had its deadline
# pass, so nothing can be done about it — and predicting only its remaining fixtures would mark
# every player whose match is already finished as a blank gameweek, which is simply wrong.
upcoming = first_kickoff[first_kickoff > pd.Timestamp.now(tz="UTC")]
TARGET_GWS = {h: int(g) for h, g in zip(HORIZONS, upcoming.index[:len(HORIZONS)])}
print("Prediction targets:")
for h, g in TARGET_GWS.items():
    print(f"  h={h} -> gw_id {g} (first kickoff {upcoming.loc[g]:%Y-%m-%d %H:%M} UTC)")

if not TARGET_GWS:
    # Normal at the end of a season and through the summer, so exit 0 — a red CI run here would
    # be noise, not a signal. SystemExit with a string exits 1; pass an int to exit cleanly.
    print("No upcoming fixtures found — nothing to predict.")
    raise SystemExit(0)

# %% [markdown]
# ## Skip if nothing has changed
#
# Predictions only move when a new gameweek's results land. `02_fpl_api.py` ingests a gameweek
# only once the FPL API flags it `finished`, which happens after bonus points are confirmed, so
# a change in `data_through_gw_id` is a reliable "there is genuinely new information" signal.
#
# Everything below is deterministic given the same inputs, so re-running on unchanged data
# rewrites identical bytes for ~4 minutes of compute. Exit instead.

# %%
FORCE = os.environ.get("FPL_FORCE_PREDICT", "").strip().lower() in ("1", "true", "yes")

if not FORCE and os.path.exists(FACT_PLAYER_POINTS_PREDICTION):
    _prev = pd.read_csv(FACT_PLAYER_POINTS_PREDICTION, usecols=["gw_id", "data_through_gw_id"])
    _same_data = int(_prev["data_through_gw_id"].max()) == DATA_THROUGH
    _same_targets = set(_prev["gw_id"].unique()) == set(TARGET_GWS.values())
    if _same_data and _same_targets:
        print(f"No new results since the last run (data still through gw {DATA_THROUGH}) and the "
              f"target gameweeks are unchanged — nothing to recompute. "
              f"Set FPL_FORCE_PREDICT=1 to override.")
        raise SystemExit(0)   # an int, so this is a clean exit rather than a failed CI step
    print(f"Rebuilding: new results={not _same_data}, targets changed={not _same_targets}")

# Early in a season the rolling windows are empty and only the prior-season features fire, so
# predictions are close to worthless. Say so loudly rather than publishing them silently.
if DATA_THROUGH % 100 < 2:
    print(f"WARNING — only {DATA_THROUGH % 100} gameweek(s) of {SEASON} are complete. "
          "Form features are mostly empty and these predictions lean almost entirely on last "
          "season. Treat them with scepticism.")

# %% [markdown]
# ## Fit and predict, one model set per horizon
#
# Separate fits per horizon because the feature shift depth differs: at h=3 the most recent
# observed result is three fixtures old, and the training matrix must reflect that.

# %%
def training_pool(df):
    """Rows with a trustworthy label.

    Double and triple gameweeks are excluded: 02_fpl_api.py upserts on (gw_id, player_id), so
    only one of a player's fixtures survives and the label understates their real haul with no
    way to tell which fixture it came from. ~4% of rows, all known-wrong.
    """
    return df[
        df["played"].fillna(False)
        & df["total_points"].notna()
        & (df["season"] >= MIN_TRAIN_SEASON)
        & (df["fixtures_in_gw"] == 1)
    ]


fixture_predictions = []
for h in HORIZONS:
    target_gw = TARGET_GWS.get(h)
    if target_gw is None:
        continue

    feats = add_baselines(add_player_features(frame, h, spine))
    FEATURES = feature_columns(feats)          # raises if a leaky column slipped in
    train = training_pool(feats)
    predict_rows = feats[(feats["gw_id"] == target_gw) & (~feats["played"].fillna(False))]

    if predict_rows.empty:
        print(f"h={h}: no rows to predict for gw {target_gw} — skipping.")
        continue

    block = predict_rows[[
        "player_id", "team_id", "position_id", "gw_id", "match_id", "is_home", "own_difficulty",
    ]].copy()
    block["horizon"] = h

    for kind, model_name in KIND_TO_MODEL.items():
        fitted = fit_two_stage(train, FEATURES, kind)
        out = predict_two_stage(fitted, predict_rows)

        # Stage 2 is run twice, once assuming a start and once assuming a cameo. That treats a
        # correlational feature as if it were causal, which usually behaves but is worth checking:
        # starters should be worth more than substitutes.
        if out["ep_rest_60"].mean() <= out["ep_rest_cameo"].mean():
            print(f"WARNING — {model_name} h={h}: predicted points given a start "
                  f"({out['ep_rest_60'].mean():.2f}) are not above those given a cameo "
                  f"({out['ep_rest_cameo'].mean():.2f}). The is_60plus decomposition is not "
                  "behaving; see fit_two_stage.")

        b = block.copy()
        b["model_name"] = model_name
        b[["p_any", "p_60", "expected_minutes", "predicted_points"]] = out[
            ["p_any", "p_60", "expected_minutes", "predicted_points"]
        ].values
        fixture_predictions.append(b)

    for baseline in ("baseline_zero", "baseline_season_mean", "baseline_last5",
                     "baseline_pp90_minutes"):
        b = block.copy()
        b["model_name"] = baseline
        b["predicted_points"] = predict_rows[baseline].values
        b[["p_any", "p_60", "expected_minutes"]] = np.nan
        fixture_predictions.append(b)

    print(f"h={h} gw {target_gw}: {len(predict_rows)} player-fixtures, "
          f"trained on {len(train):,} rows")

per_fixture = pd.concat(fixture_predictions, ignore_index=True)

# %%
# The blend is the mean of the three fitted models, computed per player-fixture.
blend = (
    per_fixture[per_fixture["model_name"].isin(BLEND_MEMBERS)]
    .groupby(["player_id", "gw_id", "horizon", "match_id", "team_id", "position_id",
              "is_home", "own_difficulty"], as_index=False)
    .agg(predicted_points=("predicted_points", "mean"), p_any=("p_any", "mean"),
         p_60=("p_60", "mean"), expected_minutes=("expected_minutes", "mean"))
)
blend["model_name"] = PRIMARY_MODEL
per_fixture = pd.concat([per_fixture, blend], ignore_index=True)

# %% [markdown]
# ## Aggregate fixtures to gameweeks
#
# Predictions are made per fixture and **summed** to the gameweek. A player with a double
# gameweek therefore gets roughly twice the points, which is correct — and is something the
# published `fact_fpl_player_gw` cannot currently express, since its upsert collapses double
# gameweeks to a single row.

# %%
predictions = per_fixture.groupby(
    ["model_name", "player_id", "gw_id", "horizon"], as_index=False
).agg(
    team_id=("team_id", "first"),
    position_id=("position_id", "first"),
    fixture_count=("match_id", "nunique"),
    mean_difficulty=("own_difficulty", "mean"),
    home_fixture_count=("is_home", "sum"),
    p_any=("p_any", "mean"),
    p_60=("p_60", "mean"),
    # min_count=1 so the baselines, which have no stage-1 output, stay null rather than summing
    # an all-null group to a misleading 0.0.
    expected_minutes=("expected_minutes", lambda s: s.sum(min_count=1)),
    predicted_points=("predicted_points", "sum"),
)

# Players whose club has no fixture in a target gameweek get an explicit zero row rather than
# being absent, so Power BI visuals do not silently drop them.
roster = predictions[["model_name", "player_id", "team_id", "position_id"]].drop_duplicates()
grid = roster.merge(pd.DataFrame({"horizon": list(TARGET_GWS), "gw_id": list(TARGET_GWS.values())}),
                    how="cross")
predictions = grid.merge(
    predictions, on=["model_name", "player_id", "gw_id", "horizon", "team_id", "position_id"],
    how="left",
)
blanks = predictions["predicted_points"].isna()
predictions.loc[blanks, ["predicted_points", "fixture_count", "home_fixture_count"]] = 0
predictions["has_fixture"] = (predictions["fixture_count"] > 0).astype(int)
predictions["predicted_at"] = RUN_DATE
predictions["data_through_gw_id"] = DATA_THROUGH

COLS = ["model_name", "player_id", "gw_id", "horizon", "team_id", "position_id",
        "fixture_count", "has_fixture", "mean_difficulty", "home_fixture_count",
        "p_any", "p_60", "expected_minutes", "predicted_points",
        "predicted_at", "data_through_gw_id"]
predictions = predictions[COLS].sort_values(
    ["model_name", "horizon", "predicted_points"], ascending=[True, True, False]
).reset_index(drop=True)

# Round before writing. This file is rebuilt and committed daily, and full float64 repr costs
# ~1.9 MB per run against ~700 KB rounded — roughly a gigabyte of git history a year saved for
# precision far below the model's actual resolution.
predictions["predicted_points"] = predictions["predicted_points"].round(3)
predictions[["p_any", "p_60"]] = predictions[["p_any", "p_60"]].round(4)
predictions[["expected_minutes", "mean_difficulty"]] = predictions[
    ["expected_minutes", "mean_difficulty"]
].round(2)

print(f"{len(predictions):,} prediction rows "
      f"({predictions['model_name'].nunique()} models x "
      f"{predictions['player_id'].nunique()} players x {len(TARGET_GWS)} gameweeks)")
print(f"Blank-gameweek rows (zeroed): {int(blanks.sum())}")

# %%
# Sanity checks before anything is written.
assert predictions["predicted_points"].notna().all(), "NaN predicted_points"
assert (predictions["fixture_count"] <= 3).all(), "More than 3 fixtures in a gameweek"
_primary = predictions[(predictions["model_name"] == PRIMARY_MODEL) & (predictions["horizon"] == 1)]
print(f"\n{PRIMARY_MODEL} h=1: mean {_primary['predicted_points'].mean():.2f}, "
      f"max {_primary['predicted_points'].max():.2f}")

# A feature that has gone all-null upstream would not raise anywhere, so compare coverage on the
# prediction rows against the training rows and complain if it has collapsed.
_f1 = add_baselines(add_player_features(frame, 1, spine))
_cols = feature_columns(_f1)
_tr_cov = training_pool(_f1)[_cols].notna().mean()
_pr_cov = _f1[_f1["gw_id"] == TARGET_GWS[1]][_cols].notna().mean()
_drop = (_tr_cov - _pr_cov).sort_values(ascending=False)
_bad = _drop[_drop > 0.20]
if len(_bad):
    print(f"WARNING — feature coverage dropped >20pp versus training for: {dict(_bad.round(2))}. "
          "An upstream column may have gone null; predictions will still be produced.")
else:
    print("Feature coverage on prediction rows is consistent with training.")

# %% [markdown]
# ## Write the datasets

# %%
predictions.to_csv(FACT_PLAYER_POINTS_PREDICTION, index=False)
print(f"Wrote {FACT_PLAYER_POINTS_PREDICTION}  ({len(predictions)} rows, "
      f"{len(predictions.columns)} cols)")

MODEL_DIM.to_csv(DIM_MODEL, index=False)
print(f"Wrote {DIM_MODEL}  ({len(MODEL_DIM)} rows)")

# An append-only record of what was predicted before each gameweek was played. The snapshot above
# is overwritten every run, so without this the dashboard could never show how accurate the model
# actually turned out to be.
history = predictions[
    (predictions["horizon"] == 1) & (predictions["model_name"] == PRIMARY_MODEL)
].copy()
upsert_csv(history, FACT_PLAYER_PREDICTION_HISTORY,
           keys=["model_name", "player_id", "gw_id"], columns=COLS)

# %% [markdown]
# ## Excel export
#
# Pivoted from the long table above rather than recomputed, so there is exactly one source of
# truth. This is a human deliverable, not a dataset, so it lives in `exports/` and leaves
# `FPL_DATA/` as pure CSV.

# %%
players = pd.read_csv(DIM_PLAYER)[["player_id", "web_name"]]
teams = pd.read_csv(DIM_TEAM)
positions = pd.read_csv(DIM_POSITION)
pos_col = [c for c in positions.columns if c != "position_id"][0]
team_col = [c for c in teams.columns if c != "team_id"][0]

fixture_labels = (
    per_fixture[per_fixture["model_name"] == PRIMARY_MODEL]
    .merge(spine[["match_id", "team_id", "opp_team_id"]].drop_duplicates(),
           on=["match_id", "team_id"], how="left")
    .merge(teams.rename(columns={"team_id": "opp_team_id", team_col: "opp"}),
           on="opp_team_id", how="left")
)
fixture_labels["label"] = np.where(
    fixture_labels["is_home"] == 1,
    fixture_labels["opp"].str.upper() + " (H)",
    fixture_labels["opp"].str.lower() + " (A)",
)
opponents = (fixture_labels.sort_values("match_id")
             .groupby(["player_id", "gw_id"])["label"]
             .apply(lambda s: ", ".join(s.dropna())).rename("opponents").reset_index())


def wide_sheet(horizon):
    sel = predictions[predictions["horizon"] == horizon]
    wide = sel.pivot_table(index="player_id", columns="model_name", values="predicted_points")
    meta = (sel[sel["model_name"] == PRIMARY_MODEL]
            [["player_id", "gw_id", "team_id", "position_id", "fixture_count",
              "mean_difficulty", "p_60", "expected_minutes"]])
    out = (meta.merge(players, on="player_id", how="left")
               .merge(teams.rename(columns={team_col: "team"}), on="team_id", how="left")
               .merge(positions.rename(columns={pos_col: "position"}), on="position_id", how="left")
               .merge(opponents, on=["player_id", "gw_id"], how="left")
               .merge(wide, on="player_id", how="left"))
    fitted_cols = [c for c in KIND_TO_MODEL.values() if c in out.columns]
    out["model_spread"] = out[fitted_cols].max(axis=1) - out[fitted_cols].min(axis=1)
    cols = (["player_id", "web_name", "team", "position", "opponents", "fixture_count",
             "mean_difficulty", "p_60", "expected_minutes"]
            + [c for c in MODEL_DIM["model_name"] if c in out.columns] + ["model_spread"])
    return out[cols].sort_values(PRIMARY_MODEL, ascending=False).reset_index(drop=True)


next_gw = wide_sheet(1)
next_gw_meta = next_gw[["player_id", "team", "position"]]
next_gw = next_gw.drop(columns="player_id")

planner = (predictions[predictions["model_name"] == PRIMARY_MODEL]
           .pivot_table(index="player_id", columns="horizon", values="predicted_points"))
horizon_cols = [f"gw+{c}" for c in planner.columns]
planner.columns = horizon_cols
planner["total_3gw"] = planner[horizon_cols].sum(axis=1)
# Join on player_id, never on web_name — web_name is not unique in dim_player and merging on it
# silently fans the sheet out beyond one row per player.
planner = (planner.reset_index()
           .merge(players, on="player_id", how="left")
           .merge(next_gw_meta, on="player_id", how="left")
           [["web_name", "team", "position"] + horizon_cols + ["total_3gw"]]
           .sort_values("total_3gw", ascending=False).reset_index(drop=True))

about = pd.DataFrame({
    "field": ["Generated (UTC)", "Data observed through", "Predicting gameweeks",
              "Primary model", "Training window", "Training rows", "Players",
              "Double gameweeks", "Blank gameweeks", "Injury and suspension news",
              "Accuracy", "Source"],
    "value": [
        RUN_AT, DATA_THROUGH, ", ".join(str(g) for g in TARGET_GWS.values()),
        PRIMARY_MODEL, f"season {MIN_TRAIN_SEASON} onwards", f"{len(training_pool(_f1)):,}",
        f"{predictions['player_id'].nunique()}",
        "Summed across both fixtures, so a double gameweek shows roughly double points.",
        "Shown as 0.0 with fixture_count 0.",
        "NOT included. The model infers availability from minutes history only, so it will "
        "confidently predict a full game for a player injured in training this week.",
        "Ranking, not precision. Single-gameweek points are close to irreducibly noisy; treat "
        "these as an ordering of who is likely to do well, not a forecast of the exact score.",
        "https://github.com/FabianMyrvang/fpl-analytics",
    ],
})

os.makedirs(EXPORT_DIR, exist_ok=True)
with pd.ExcelWriter(XLSX_PREDICTIONS, engine="openpyxl") as xl:
    next_gw.to_excel(xl, sheet_name="Next GW", index=False)
    planner.to_excel(xl, sheet_name="Next 3 GW", index=False)
    MODEL_DIM.to_excel(xl, sheet_name="Models", index=False)
    about.to_excel(xl, sheet_name="About", index=False)

    for name, df in (("Next GW", next_gw), ("Next 3 GW", planner),
                     ("Models", MODEL_DIM), ("About", about)):
        ws = xl.sheets[name]
        ws.freeze_panes = "A2"
        for i, col in enumerate(df.columns, start=1):
            width = max(len(str(col)), int(df[col].astype(str).str.len().max() or 0))
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(width + 2, 55)
            if df[col].dtype.kind == "f":
                for row in range(2, len(df) + 2):
                    ws.cell(row=row, column=i).number_format = "0.0"

print(f"Wrote {XLSX_PREDICTIONS}")

# %%
print(f"\nTop 15 predicted for gw {TARGET_GWS[1]} ({PRIMARY_MODEL}):")
print(next_gw.head(15).to_string(index=False))
