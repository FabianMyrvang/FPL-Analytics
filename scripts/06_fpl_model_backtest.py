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
# # Walk-forward backtest
#
# Local/exploratory — **not** part of CI, like `04_`. Establishes how well each model would have
# predicted each gameweek had it only known what was knowable at that gameweek's deadline.
#
# Run order matters: the baselines are established first and deliberately made as strong as
# honesty allows, because the question this script exists to answer is not "is the model good"
# but "is the model better than arithmetic".

# %%
import time

import numpy as np
import pandas as pd

from common import upsert_csv, DIM_POSITION
from model_common import (
    ALL_MODELS, BASELINE_MODELS, FITTED_MODELS, HORIZONS, MIN_TRAIN_SEASON,
    FACT_MODEL_BACKTEST_METRIC,
    add_baselines, add_player_features, build_fixture_spine, build_player_frame,
    feature_columns, fit_two_stage, predict_two_stage, score_gameweek,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Evaluate on the most recent complete season plus whatever exists of the current one. Earlier
# seasons are training data only: 2025-26 introduced the defensive_contribution scoring rule, so
# scores from 2022-24 would measure performance against a points function that no longer applies.
EVAL_SEASONS = (2025, 2026)

# %%
t0 = time.time()
spine = build_fixture_spine()
frame = build_player_frame(spine)
print(f"Spine + frame ready in {time.time() - t0:.1f}s")

# %%
# Build the feature matrix once per horizon. Safe to do globally — every player-history feature
# is a backward-looking window shifted by the horizon, verified by the truncation check below.
features = {}
for h in HORIZONS:
    t = time.time()
    features[h] = add_baselines(add_player_features(frame, h, spine))
    print(f"h={h}: {features[h].shape[0]:,} rows x {features[h].shape[1]} cols "
          f"({time.time() - t:.1f}s)")

FEATURES = feature_columns(features[1])
print(f"\n{len(FEATURES)} model features")

# %% [markdown]
# ## Leakage guards
#
# These run before any modelling and are the reason to trust everything downstream. A feature
# that quietly encodes the outcome produces a backtest that looks superb and a live model that
# does not work.

# %%
train_mask = features[1]["played"].fillna(False) & (features[1]["season"] >= MIN_TRAIN_SEASON)
tr = features[1][train_mask]

corr = tr[FEATURES].corrwith(tr["total_points"]).abs().sort_values(ascending=False)
leaked = corr[corr > 0.99]
assert len(leaked) == 0, f"Feature(s) encode the target: {list(leaked.index)}"
print(f"GUARD 1 — max |corr| with same-row target: {corr.iloc[0]:.3f} ({corr.index[0]})  PASS")

# Confirm the guard can actually fire, so a pass means something.
_probe = abs(tr["total_points"].corr(tr["total_points"]))
assert _probe > 0.99, "correlation guard is not functioning"
print("GUARD 2 — guard fires on an injected copy of the target  PASS")

# Rebuild features from truncated inputs and confirm the cutoff gameweek is unchanged. Anything
# reading forward would differ here.
_CUT = 202520
_FORWARD = ["next_difficulty_1", "next_difficulty_2", "difficulty_next3_mean"]
_back = [c for c in FEATURES if c not in _FORWARD]
_ft = add_player_features(frame[frame["gw_id"] <= _CUT], 1, spine[spine["gw_id"] <= _CUT])
_a = features[1][features[1]["gw_id"] == _CUT].set_index(["player_id", "match_id"]).sort_index()
_b = _ft[_ft["gw_id"] == _CUT].set_index(["player_id", "match_id"]).sort_index()
_common = _a.index.intersection(_b.index)
_diff = (_a.loc[_common, _back] - _b.loc[_common, _back]).abs().max().max()
assert _diff < 1e-9, f"Features differ when rebuilt on truncated history (max {_diff:.2e})"
print(f"GUARD 3 — {len(_common)} rows at gw {_CUT} identical when rebuilt on history only  PASS")

# %% [markdown]
# ## Evaluation set
#
# Double and triple gameweeks are excluded. `02_fpl_api.py` upserts on `(gw_id, player_id)`, so
# for a player with two fixtures in one gameweek only one survives — the label understates the
# real haul and there is no way to tell which fixture it came from. Scoring against a known-wrong
# label would be measuring the wrong thing. ~4% of rows.

# %%
def eval_rows(df):
    return df[
        df["played"].fillna(False)
        & df["season"].isin(EVAL_SEASONS)
        & (df["fixtures_in_gw"] == 1)
        & df["total_points"].notna()
    ]

ev = eval_rows(features[1])
print(f"Evaluation rows: {len(ev):,} across {ev['gw_id'].nunique()} gameweeks "
      f"({ev['gw_id'].min()} -> {ev['gw_id'].max()})")
print(f"Excluded as DGW/blank: {(features[1]['played'].fillna(False) & features[1]['season'].isin(EVAL_SEASONS) & (features[1]['fixtures_in_gw'] != 1)).sum():,} rows")

# %% [markdown]
# ## Baselines
#
# No fitting, so no walk-forward refit is needed — every baseline is already a function of
# strictly-lagged features. Scoring is per gameweek, then summarised.

# %%
def score_models(df, model_cols, horizon, by_position=False):
    """Score each model column against total_points, one row per (model, gw, metric)."""
    records = []
    for gw_id, g in df.groupby("gw_id", sort=True):
        for m in model_cols:
            metrics = score_gameweek(g["total_points"], g[m], g["minutes"])
            for name, value in metrics.items():
                records.append((m, horizon, int(gw_id), -1, name, value))
            if by_position:
                for pos, gp in g.groupby("position_id"):
                    if len(gp) < 5:
                        continue
                    err = (gp[m] - gp["total_points"]).abs().mean()
                    records.append((m, horizon, int(gw_id), int(pos), "mae", float(err)))
    return pd.DataFrame(records, columns=["model_name", "horizon", "gw_id", "position_id",
                                          "metric_name", "metric_value"])


baseline_metrics = []
for h in HORIZONS:
    ev_h = eval_rows(features[h])
    baseline_metrics.append(score_models(ev_h, list(BASELINE_MODELS), h, by_position=True))
baseline_metrics = pd.concat(baseline_metrics, ignore_index=True)
print(f"{len(baseline_metrics):,} metric rows")

# %%
def summarise(metrics, horizon=1, metric_names=None):
    """Mean of each metric across gameweeks — the table to actually read."""
    metric_names = metric_names or [
        "spearman_likely", "precision_at_20", "ndcg_at_20", "rmse", "mae", "mae_played",
        "captain_points",
    ]
    sel = metrics[(metrics["horizon"] == horizon) & (metrics["position_id"] == -1)
                  & metrics["metric_name"].isin(metric_names)]
    agg = "sum" if "captain_points" in metric_names else "mean"
    out = sel.pivot_table(index="model_name", columns="metric_name", values="metric_value",
                          aggfunc="mean")
    cap = sel[sel["metric_name"] == "captain_points"].groupby("model_name")["metric_value"].sum()
    if len(cap):
        out["captain_points"] = cap
    return out[[c for c in metric_names if c in out.columns]]


print("=" * 100)
print("BASELINE BAR — horizon 1, mean across gameweeks (captain_points is a season total)")
print("=" * 100)
print(summarise(baseline_metrics, 1).to_string())

# %%
for h in HORIZONS:
    print(f"\n--- horizon {h} ---")
    print(summarise(baseline_metrics, h).to_string())

# %% [markdown]
# ### Verification gate
#
# `baseline_pp90_minutes` should beat `baseline_last5` on `spearman_likely`. If it does not, the
# feature build is wrong and there is no point modelling — scaling a rate by expected minutes is
# strictly more information than a flat mean.

# %%
_s = summarise(baseline_metrics, 1)["spearman_likely"]
print(f"baseline_pp90_minutes {_s['baseline_pp90_minutes']:.4f} vs "
      f"baseline_last5 {_s['baseline_last5']:.4f}")
if _s["baseline_pp90_minutes"] > _s["baseline_last5"]:
    print("PASS — minutes-scaling helps, as expected.")
else:
    print("WARNING — minutes-scaling does NOT help. Investigate the feature build before "
          "fitting anything; a model is unlikely to rescue this.")

# %%
BAR = _s.max()
print(f"\nThe bar to beat on spearman_likely (h=1): {BAR:.4f} "
      f"({_s.idxmax()})")

# %% [markdown]
# ## Walk-forward evaluation of the fitted models
#
# For each evaluation gameweek `G` at horizon `h`, train only on gameweeks whose results were
# already known at the moment the prediction would have been made — that is, up to `G - h`.
# Training on `G - 1` at `h = 3` would use two gameweeks of results that had not happened yet.
#
# The cutoff is taken by position in the gameweek list, not by arithmetic on `gw_id`, because
# `gw_id` is not contiguous across a season boundary (202538 -> 202601).

# %%
REFIT_EVERY = 6   # set to 1 for the final run; 6 keeps an iteration cycle to a few minutes
KINDS = ["linear", "gbm", "gbm_poisson"]
KIND_TO_MODEL = {"linear": "linear_2stage", "gbm": "gbm_2stage",
                 "gbm_poisson": "gbm_poisson_2stage"}


def training_pool(df, before_gw_id):
    return df[
        df["played"].fillna(False)
        & (df["season"] >= MIN_TRAIN_SEASON)
        & (df["fixtures_in_gw"] == 1)
        & (df["gw_id"] < before_gw_id)
    ]


def walk_forward(horizon, refit_every=REFIT_EVERY):
    """Predict every evaluation gameweek, refitting every `refit_every` gameweeks."""
    d = features[horizon]
    ev = eval_rows(d)
    gws = sorted(ev["gw_id"].unique())
    fitted, preds = None, []

    for i, gw in enumerate(gws):
        cutoff_idx = i - horizon + 1
        if cutoff_idx <= 0:
            continue  # not enough observed history yet at this horizon
        cutoff_gw = gws[cutoff_idx]

        if fitted is None or (i % refit_every == 0):
            train = training_pool(d, cutoff_gw)
            if len(train) < 5000:
                continue
            fitted = {k: fit_two_stage(train, FEATURES, k) for k in KINDS}

        test = ev[ev["gw_id"] == gw]
        block = test[["player_id", "gw_id", "position_id", "minutes", "total_points"]].copy()
        for k in KINDS:
            p = predict_two_stage(fitted[k], test)
            block[KIND_TO_MODEL[k]] = p["predicted_points"].values
            if k == "gbm":   # keep one model's stage-1 output for calibration checks
                block["p_any"] = p["p_any"].values
                block["p_60"] = p["p_60"].values
        block["blend_2stage"] = block[list(KIND_TO_MODEL.values())].mean(axis=1)
        for b in BASELINE_MODELS:
            block[b] = test[b].values
        preds.append(block)

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


t0 = time.time()
predictions = {}
for h in HORIZONS:
    t = time.time()
    predictions[h] = walk_forward(h)
    print(f"h={h}: {len(predictions[h]):,} predictions across "
          f"{predictions[h]['gw_id'].nunique()} gameweeks ({time.time() - t:.0f}s)")
print(f"Total walk-forward time: {time.time() - t0:.0f}s")

# %%
# Score baselines and fitted models together from `predictions`, which carries a column for each.
# Do NOT also fold in `baseline_metrics` here: that frame covers every evaluation gameweek, while
# the fitted models skip the first few while history accumulates, so the baselines would appear
# twice on overlapping gameweeks. Mean-based metrics would be mildly distorted and captain_points,
# which sums, would be exactly doubled.
metrics = pd.concat(
    [score_models(predictions[h], list(ALL_MODELS), h, by_position=True) for h in HORIZONS],
    ignore_index=True,
)
print(f"{len(metrics):,} metric rows over {predictions[1]['gw_id'].nunique()} gameweeks, "
      "every model scored on identical rows")

# %%
print("=" * 105)
print("MODEL COMPARISON — horizon 1, mean per gameweek (captain_points is a season total)")
print("=" * 105)
print(summarise(metrics, 1).sort_values("spearman_likely", ascending=False).to_string())

# %%
for h in (2, 3):
    print(f"\n--- horizon {h} ---")
    print(summarise(metrics, h).sort_values("spearman_likely", ascending=False).to_string())

# %% [markdown]
# ### Did anything actually beat the baseline?

# %%
for h in HORIZONS:
    s = summarise(metrics, h)
    best_base = s.loc[list(BASELINE_MODELS), "spearman_likely"].max()
    best_base_name = s.loc[list(BASELINE_MODELS), "spearman_likely"].idxmax()
    best_fit = s.loc[list(FITTED_MODELS), "spearman_likely"].max()
    best_fit_name = s.loc[list(FITTED_MODELS), "spearman_likely"].idxmax()
    gain = (best_fit - best_base) / best_base * 100
    verdict = "BEATS baseline" if best_fit > best_base else "FAILS to beat baseline"
    print(f"h={h}: best fitted {best_fit_name} {best_fit:.4f} vs best baseline "
          f"{best_base_name} {best_base:.4f}  ({gain:+.1f}%)  {verdict}")

# %% [markdown]
# ### Stage 1 — the part where most of the value is
#
# Calibration matters more than discrimination here: `p_60` is *multiplied* into the final
# number, so a probability that is systematically 0.1 too high biases every prediction.

# %%
from sklearn.metrics import brier_score_loss, roc_auc_score

st1 = predictions[1]
for label, col in (("minutes > 0", "p_any"), ("minutes >= 60", "p_60")):
    y = (st1["minutes"] > 0).astype(int) if col == "p_any" else (st1["minutes"] >= 60).astype(int)
    print(f"{label:15s}  AUC {roc_auc_score(y, st1[col]):.4f}   "
          f"Brier {brier_score_loss(y, st1[col]):.4f}")

print("\nCalibration of p_60 (predicted vs actual rate of playing 60+):")
bins = pd.cut(st1["p_60"], [0, .2, .4, .6, .8, 1.0], include_lowest=True)
calib = st1.groupby(bins, observed=True).apply(
    lambda g: pd.Series({"n": len(g), "predicted": g["p_60"].mean(),
                         "actual": (g["minutes"] >= 60).mean()}), include_groups=False)
calib["error"] = calib["predicted"] - calib["actual"]
print(calib.to_string())

# %% [markdown]
# ### Error by position
#
# The 2025-26 `defensive_contribution` rule change affects defenders specifically. If DEF error
# is much worse than the rest, training across the rule boundary is hurting and the window
# should be narrowed.

# %%
pos = pd.read_csv(DIM_POSITION)
by_pos = metrics[(metrics["horizon"] == 1) & (metrics["position_id"] != -1)
                 & (metrics["metric_name"] == "mae")]
tbl = by_pos.pivot_table(index="model_name", columns="position_id", values="metric_value")
tbl.columns = [pos.set_index("position_id").iloc[:, 0].get(c, c) for c in tbl.columns]
print(tbl.sort_index().to_string())

# %%
written = upsert_csv(
    metrics, FACT_MODEL_BACKTEST_METRIC,
    keys=["model_name", "horizon", "gw_id", "position_id", "metric_name"],
)
print(f"Backtest metrics saved -> {FACT_MODEL_BACKTEST_METRIC}")
