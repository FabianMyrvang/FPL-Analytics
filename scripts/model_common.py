"""Feature engineering and shared helpers for the points-prediction model.

Imported by 05_fpl_predict_points.py and 06_fpl_model_backtest.py.

Deliberately kept SEPARATE from common.py. That module promises pandas-only so 02_ and 03_
keep running under the slim requirements-ci.txt; this one is free to import scikit-learn and
is only installed for the prediction workflow (requirements-model.txt).

The central design decision is that history and future live in ONE frame, one row per player
per fixture, sorted by kickoff time. Features are backward-looking rolling windows shifted by
the prediction horizon, so the same code path produces training rows and prediction rows and
train/serve skew is impossible by construction. `fact_fpl_fixture.csv` carries FDR for every
fixture of the season including the unplayed ones, which is what makes that possible.
"""

import numpy as np
import pandas as pd

from common import (
    DATA_DIR,
    DIM_FIXTURE,
    FACT_FPL_FIXTURE,
    FACT_FPL_PLAYER_GW,
)

# --- Outputs -------------------------------------------------------------------------------
FACT_PLAYER_POINTS_PREDICTION = f"{DATA_DIR}/fact_player_points_prediction.csv"
FACT_PLAYER_PREDICTION_HISTORY = f"{DATA_DIR}/fact_player_points_prediction_history.csv"
FACT_MODEL_BACKTEST_METRIC = f"{DATA_DIR}/fact_model_backtest_metric.csv"
DIM_MODEL = f"{DATA_DIR}/dim_model.csv"

EXPORT_DIR = "exports"
XLSX_PREDICTIONS = f"{EXPORT_DIR}/fpl_points_predictions.xlsx"


# --- Modelling constants -------------------------------------------------------------------

# Training starts here. expected_goals/_assists/_goal_involvements/_goals_conceded and `starts`
# are 100% null before 2022-23, so the two earlier seasons would contribute ~47k rows that force
# five of the strongest features to NaN and teach a tree model to split on "is this the old era".
MIN_TRAIN_SEASON = 2022

# 2025-26 introduced `defensive_contribution` as a SCORING RULE, so a 2023-24 defender's
# total_points is drawn from a different points function than today's. Down-weighting earlier
# seasons absorbs some of that regime shift without discarding 80k rows. 06_ sweeps this.
PRE_DC_SEASON_WEIGHT = 0.6
DC_RULE_SEASON = 2025

HORIZONS = (1, 2, 3)

# Never let these reach a model. All four are FROZEN_SNAPSHOT_COLS in 02_fpl_api.py: the FPL API
# exposes only their current value, so the pipeline captures them at an arbitrary point INSIDE the
# gameweek window. transfers_in/out are the dangerous ones — they move on press-conference injury
# news, which is exactly what stage 1 is trying to predict. Including them produces a model that
# backtests beautifully and fails live. `now_cost` is deliberately NOT here: it drifts ±0.1 per GW
# so the contamination is negligible, and it is a good proxy for underlying quality.
LEAKY_COLS = (
    "selected_by_percent",
    "transfers_in",
    "transfers_out",
)

# Post-match columns, safe only as lagged rolling aggregates and never at their own row.
_FORM_W5 = [
    "total_points", "minutes", "bps", "ict_index", "influence", "creativity", "threat",
    "goals_scored", "assists", "bonus", "clean_sheets", "saves", "expected_goals",
    "expected_assists", "expected_goal_involvements", "expected_goals_conceded", "yellow_cards",
]
_FORM_W3 = ["total_points", "minutes", "bps", "ict_index"]
_FORM_W10 = ["total_points", "minutes", "expected_goal_involvements", "expected_goals_conceded"]

# Per-90 rates: rolling_sum(x, 5) / rolling_sum(minutes, 5) * 90.
_PER90 = ["total_points", "bps", "expected_goals", "expected_assists", "expected_goal_involvements"]

# Below this many minutes in the window a per-90 rate is noise, not signal — a 12-minute cameo
# with a tap-in becomes 15 xG per 90 and Ridge will chase it. Emit NaN instead.
_PER90_MIN_MINUTES = 90


def season_of(gw_id):
    """Season start year from a gw_id (202601 -> 2026). Works on scalars and Series."""
    return gw_id // 100


def gw_of(gw_id):
    """Gameweek number from a gw_id (202601 -> 1)."""
    return gw_id % 100


# --- Fixture spine -------------------------------------------------------------------------

def build_fixture_spine():
    """One row per (team_id, match_id) for every fixture, played and unplayed.

    fact_fpl_fixture carries team_h_difficulty/team_a_difficulty for all 380 fixtures of the
    season, not just the completed ones, and dim_fixture carries kickoff_time for all of them.
    That is what lets training and prediction share this function.

    Returns columns: team_id, opp_team_id, match_id, gw_id, season, gw, kickoff_time, is_home,
    own_difficulty, opp_difficulty, goals_for, goals_against, played, fixture_seq, rest_days,
    fixtures_in_gw, next_difficulty_1, next_difficulty_2, difficulty_next3_mean.
    """
    fixtures = pd.read_csv(DIM_FIXTURE)
    fdr = pd.read_csv(FACT_FPL_FIXTURE)

    keep = ["match_id", "team_h_difficulty", "team_a_difficulty", "home_score", "away_score"]
    merged = fixtures.merge(fdr[keep], on="match_id", how="left")
    merged["kickoff_time"] = pd.to_datetime(merged["kickoff_time"], utc=True, errors="coerce")

    # Unpivot home/away into one row per team per match. Note team_h_difficulty is the difficulty
    # of the fixture FOR the home team, so it is the home row's own_difficulty.
    home = merged.assign(
        team_id=merged["home_team_id"], opp_team_id=merged["away_team_id"], is_home=1,
        own_difficulty=merged["team_h_difficulty"], opp_difficulty=merged["team_a_difficulty"],
        goals_for=merged["home_score"], goals_against=merged["away_score"],
    )
    away = merged.assign(
        team_id=merged["away_team_id"], opp_team_id=merged["home_team_id"], is_home=0,
        own_difficulty=merged["team_a_difficulty"], opp_difficulty=merged["team_h_difficulty"],
        goals_for=merged["away_score"], goals_against=merged["home_score"],
    )
    cols = ["team_id", "opp_team_id", "match_id", "gw_id", "kickoff_time", "is_home",
            "own_difficulty", "opp_difficulty", "goals_for", "goals_against"]
    spine = pd.concat([home[cols], away[cols]], ignore_index=True)

    spine["season"] = season_of(spine["gw_id"])
    spine["gw"] = gw_of(spine["gw_id"])
    spine["played"] = spine["goals_for"].notna()
    spine = spine.sort_values(["team_id", "season", "kickoff_time", "match_id"]).reset_index(drop=True)

    grp = spine.groupby(["team_id", "season"], sort=False)
    spine["fixture_seq"] = grp.cumcount() + 1
    spine["rest_days"] = (
        grp["kickoff_time"].diff().dt.total_seconds().div(86400).clip(1, 14)
    )
    spine["fixtures_in_gw"] = spine.groupby(["team_id", "gw_id"])["match_id"].transform("size")

    # Forward-looking, but legitimately so: the fixture list is published months in advance.
    # A hard run coming up is a real rotation signal.
    spine["next_difficulty_1"] = grp["own_difficulty"].shift(-1)
    spine["next_difficulty_2"] = grp["own_difficulty"].shift(-2)
    spine["difficulty_next3_mean"] = spine[
        ["next_difficulty_1", "next_difficulty_2"]
    ].join(grp["own_difficulty"].shift(-3).rename("_d3")).mean(axis=1)

    # Triple gameweeks are rare but real — Man Utd played three times in 2020-21 GW35 after the
    # COVID fixture pile-up (the only instance in seven seasons). Nothing downstream assumes two:
    # training drops every fixtures_in_gw > 1 row, and prediction sums per fixture. Worth a line
    # in the log so a genuinely novel schedule is visible rather than silent.
    if (spine["fixtures_in_gw"] > 2).any():
        bad = spine.loc[spine["fixtures_in_gw"] > 2, ["team_id", "gw_id", "fixtures_in_gw"]]
        print(f"NOTE — {len(bad.drop_duplicates())} team-gameweek(s) with 3+ fixtures:\n"
              f"{bad.drop_duplicates().to_string(index=False)}")
    return spine


def add_team_form(spine, horizon, window=5):
    """Rolling team strength as of `horizon` fixtures back, per (team_id, match_id).

    Shifted first then rolled, which is equivalent to rolling then shifting but lets the shift
    run vectorised over the whole column.
    """
    df = spine.sort_values(["team_id", "season", "kickoff_time", "match_id"]).copy()
    df["clean_sheet"] = (df["goals_against"] == 0).where(df["played"])

    keys = ["team_id", "season"]
    src = ["goals_for", "goals_against", "clean_sheet"]
    shifted = df.groupby(keys, sort=False)[src].shift(horizon)
    rolled = (
        shifted.groupby([df["team_id"], df["season"]], sort=False)
        .rolling(window, min_periods=1).mean()
        .droplevel([0, 1]).sort_index()
    )
    out = df[["team_id", "match_id"]].copy()
    out[f"team_gf_{window}"] = rolled["goals_for"].values
    out[f"team_ga_{window}"] = rolled["goals_against"].values
    out[f"team_cs_rate_{window}"] = rolled["clean_sheet"].values
    return out


# --- Player frame --------------------------------------------------------------------------

def build_player_frame(spine, season=None):
    """One row per (player_id, match_id) for every played AND upcoming fixture.

    History comes from fact_fpl_player_gw. Upcoming rows are each currently-active player
    crossed with their own team's remaining fixtures, with all stat columns left NaN — so the
    rolling windows below flow across the boundary without any special-casing.

    `season` pins which season's remaining fixtures to project (defaults to the latest in the
    player fact).
    """
    hist = pd.read_csv(FACT_FPL_PLAYER_GW)
    hist["season"] = season_of(hist["gw_id"])
    current_season = season if season is not None else int(hist["season"].max())

    spine_cols = ["team_id", "match_id", "opp_team_id", "gw_id", "season", "gw", "kickoff_time",
                  "is_home", "own_difficulty", "opp_difficulty", "fixture_seq", "rest_days",
                  "fixtures_in_gw", "next_difficulty_1", "next_difficulty_2",
                  "difficulty_next3_mean", "played"]

    hist = hist.merge(spine[spine_cols], on=["team_id", "match_id"], how="inner",
                      suffixes=("", "_spine"))
    for dup in ("gw_id_spine", "season_spine"):
        hist = hist.drop(columns=dup, errors="ignore")

    # Each player's current club, from their most recent appearance in the fact table. A January
    # transfer means one gameweek projected against the wrong club's fixtures — warned on below.
    latest = (
        hist[hist["season"] == current_season]
        .sort_values(["player_id", "kickoff_time"])
        .groupby("player_id", as_index=False)
        .last()[["player_id", "team_id", "position_id"]]
    )

    future_fixtures = spine.loc[
        (spine["season"] == current_season) & (~spine["played"]), spine_cols
    ]
    future = latest.merge(future_fixtures, on="team_id", how="inner")

    frame = pd.concat([hist, future], ignore_index=True)
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    frame = frame.sort_values(["player_id", "season", "kickoff_time", "match_id"]).reset_index(drop=True)

    print(f"Player frame: {len(hist):,} played rows + {len(future):,} upcoming rows "
          f"({latest['player_id'].nunique()} active players, season {current_season})")
    return frame


def _prev_season_features(frame):
    """Cold-start features for a player's first fixtures of a season."""
    played = frame[frame["played"].fillna(False)]
    agg = played.groupby(["player_id", "season"]).agg(
        _mins=("minutes", "sum"), _pts=("total_points", "sum")
    ).reset_index()
    agg["prev_season_minutes"] = agg["_mins"]
    agg["prev_season_points_per_90"] = np.where(
        agg["_mins"] >= 450, agg["_pts"] / agg["_mins"] * 90, np.nan
    )
    agg["season"] = agg["season"] + 1  # attach to the FOLLOWING season
    return agg[["player_id", "season", "prev_season_minutes", "prev_season_points_per_90"]]


def add_player_features(frame, horizon, spine):
    """Attach every model feature to `frame` for a given prediction horizon.

    THE leakage rule, applied mechanically to every player-history feature: shift by `horizon`,
    not by 1. Standing at the GW3 deadline predicting GW5, the last observed result is GW2, so
    features must be three fixtures stale. Baking the horizon into the shift is what makes h=2
    and h=3 honest rather than quietly optimistic.

    Windows are grouped by (player_id, season) so no form carries across a summer.
    """
    df = frame.sort_values(["player_id", "season", "kickoff_time", "match_id"]).reset_index(drop=True)
    keys = ["player_id", "season"]
    gkeys = [df["player_id"], df["season"]]
    grp = df.groupby(keys, sort=False)

    roll_src = sorted(set(_FORM_W5 + _FORM_W3 + _FORM_W10 + _PER90 + ["starts", "now_cost"]))
    roll_src = [c for c in roll_src if c in df.columns]
    shifted = grp[roll_src].shift(horizon)

    # Carry the last OBSERVED value forward before rolling. This is a no-op during training,
    # where every preceding fixture has been played and therefore has real values — a blank
    # gameweek is minutes=0, not a null. It matters at prediction time: the script runs daily,
    # so it frequently has to predict gameweek N+1 while gameweek N is still in progress. Without
    # this, shift(1) lands on an unplayed fixture and every lag feature returns null for exactly
    # the players whose match has not kicked off yet. Falling back to the most recent fixture
    # that actually happened is what a human would do.
    shifted = shifted.groupby(gkeys, sort=False).ffill()
    shifted_grp = shifted.groupby(gkeys, sort=False)

    feats = {}
    for window, cols in ((3, _FORM_W3), (5, _FORM_W5), (10, _FORM_W10)):
        cols = [c for c in cols if c in shifted.columns]
        rolled = (shifted_grp[cols].rolling(window, min_periods=1).mean()
                  .droplevel([0, 1]).sort_index())
        for c in cols:
            feats[f"{c}_mean_{window}"] = rolled[c].values

    # EWM form. In practice these beat flat rolling means for FPL; keep both and let the model pick.
    for c in ("total_points", "minutes"):
        feats[f"{c}_ewm"] = (
            shifted_grp[c].apply(lambda s: s.ewm(halflife=3, ignore_na=True).mean())
            .droplevel([0, 1]).sort_index().values
        )

    # Per-90 rates — the features that actually drive stage 2. Computed over two windows.
    #
    # Both windows exist because a rate and a minutes average taken over the SAME window cancel:
    #     per90_5 * minutes_mean_5 / 90  ==  [sum(pts)/sum(min)*90] * [sum(min)/5] / 90
    #                                    ==  sum(pts)/5  ==  total_points_mean_5
    # so baseline_pp90_minutes would be an exact restatement of baseline_last5 (verified — the two
    # agreed to 3.6e-15). It therefore pairs the 10-fixture rate with 3-fixture minutes, which is a
    # genuine quality x availability estimate.
    per90_src = [c for c in _PER90 if c in shifted.columns]
    for window, suffix in ((5, "per90_5"), (10, "per90_10")):
        sums = (shifted_grp[per90_src + ["minutes"]]
                .rolling(window, min_periods=1).sum().droplevel([0, 1]).sort_index())
        enough = sums["minutes"] >= _PER90_MIN_MINUTES
        for c in per90_src:
            feats[f"{c}_{suffix}"] = np.where(enough, sums[c] / sums["minutes"] * 90, np.nan)

    # Availability — the stage-1 workhorses.
    for lag in (1, 2, 3):
        feats[f"minutes_lag{lag}"] = (
            grp["minutes"].shift(horizon + lag - 1).groupby(gkeys, sort=False).ffill().values
        )
    played60 = (shifted["minutes"] >= 60).astype(float).where(shifted["minutes"].notna())
    playedany = (shifted["minutes"] > 0).astype(float).where(shifted["minutes"].notna())
    p60_grp = played60.groupby(gkeys, sort=False)
    feats["p60_rate_5"] = p60_grp.rolling(5, min_periods=1).mean().droplevel([0, 1]).sort_index().values
    feats["blank_count_5"] = (
        (1 - playedany).groupby(gkeys, sort=False).rolling(5, min_periods=1).sum()
        .droplevel([0, 1]).sort_index().values
    )
    if "starts" in shifted.columns:
        feats["start_rate_5"] = (
            shifted["starts"].groupby(gkeys, sort=False).rolling(5, min_periods=1).mean()
            .droplevel([0, 1]).sort_index().values
        )

    # Consecutive prior fixtures with 60+ minutes, capped — a "nailed on" signal.
    block = (played60.fillna(0) == 0).groupby(gkeys, sort=False).cumsum()
    feats["consec_60_streak"] = (
        played60.fillna(0).groupby([df["player_id"], df["season"], block], sort=False)
        .cumsum().clip(upper=5).values
    )

    # Days since the player last actually appeared, as known `horizon` fixtures ago.
    last_appearance = (
        df["kickoff_time"].where(shifted["minutes"] > 0).groupby(gkeys, sort=False).ffill()
    )
    feats["days_since_last_appearance"] = (
        (df["kickoff_time"] - last_appearance).dt.total_seconds().div(86400).clip(0, 60).values
    )

    # Season-to-date mean, used both as a feature and as the baseline_season_mean prediction.
    _cum_pts = shifted["total_points"].groupby(gkeys, sort=False).cumsum()
    _cum_n = shifted["total_points"].notna().groupby(gkeys, sort=False).cumsum()
    feats["total_points_expanding"] = (_cum_pts / _cum_n.replace(0, np.nan)).values

    cum_minutes = shifted["minutes"].groupby(gkeys, sort=False).cumsum()
    played_so_far = (df["fixture_seq"] - horizon).clip(lower=1)
    feats["season_minutes_share"] = (cum_minutes / (90 * played_so_far)).values

    # Price, lagged like everything else — read from the forward-filled block so an in-progress
    # gameweek does not null it out.
    feats["now_cost_lag"] = shifted["now_cost"].values

    out = df.assign(**feats)

    out["difficulty_delta"] = out["opp_difficulty"] - out["own_difficulty"]
    out["is_new_season_early"] = (out["gw"] <= 4).astype(int)

    prev = _prev_season_features(df)
    out = out.merge(prev, on=["player_id", "season"], how="left")

    team_form = add_team_form(spine, horizon)
    out = out.merge(team_form, on=["team_id", "match_id"], how="left")
    opp_form = team_form.rename(columns={"team_id": "opp_team_id"})
    opp_form = opp_form.rename(columns={c: c.replace("team_", "opp_") for c in opp_form.columns
                                        if c.startswith("team_") and c != "team_id"})
    out = out.merge(opp_form, on=["opp_team_id", "match_id"], how="left")

    return out


def feature_columns(df):
    """The model's feature list, derived from what add_player_features produced.

    Asserts no leaky column slipped in. This is the single most important guard in the module:
    the snapshot columns would inflate backtest scores and evaporate live.
    """
    static = [
        "position_id", "is_home", "own_difficulty", "opp_difficulty", "difficulty_delta",
        "rest_days", "fixtures_in_gw", "next_difficulty_1", "next_difficulty_2",
        "difficulty_next3_mean", "gw", "is_new_season_early", "now_cost_lag",
        "prev_season_minutes", "prev_season_points_per_90",
        "team_gf_5", "team_ga_5", "team_cs_rate_5", "opp_gf_5", "opp_ga_5", "opp_cs_rate_5",
        "consec_60_streak", "days_since_last_appearance", "season_minutes_share",
        "p60_rate_5", "blank_count_5", "start_rate_5",
        "minutes_lag1", "minutes_lag2", "minutes_lag3",
    ]
    derived = [c for c in df.columns
               if c.endswith(("_mean_3", "_mean_5", "_mean_10", "_ewm", "_per90_5"))
               or c == "total_points_expanding"]
    cols = [c for c in static if c in df.columns] + sorted(derived)

    leaked = [c for c in cols if any(c == bad or c.startswith(bad) for bad in LEAKY_COLS)]
    if leaked:
        raise ValueError(
            f"Leaky columns reached the feature list: {leaked}. These are FROZEN_SNAPSHOT_COLS "
            "captured mid-gameweek and must never be used — see LEAKY_COLS."
        )
    return cols


# --- Baselines -----------------------------------------------------------------------------
# Every fitted model is measured against these. `baseline_pp90_minutes` is the one that matters:
# it is a hand-rolled two-stage model with nothing fitted, so a trained model that fails to beat
# it has learned nothing that simple arithmetic did not already capture.

BASELINE_MODELS = ("baseline_zero", "baseline_season_mean", "baseline_last5",
                   "baseline_pp90_minutes")


def add_baselines(df):
    """Attach one prediction column per baseline model. Operates on add_player_features output."""
    out = df.copy()
    out["baseline_zero"] = 0.0
    out["baseline_season_mean"] = out["total_points_expanding"].fillna(0.0)
    out["baseline_last5"] = out["total_points_mean_5"].fillna(0.0)

    # Quality x availability: a long-run scoring rate scaled by recent minutes. The windows must
    # differ — using the 5-fixture rate against 5-fixture minutes cancels to total_points_mean_5
    # exactly (verified: max difference 3.6e-15), which would make this baseline a duplicate.
    # The 10-fixture rate is a steadier estimate of how good the player is; the 3-fixture minutes
    # are a fresher estimate of whether they are currently playing.
    #
    # Where the rate is NaN the player has under 90 minutes in the window, so fall back to their
    # flat mean rather than to zero — the point of a baseline is to be as strong as honesty allows.
    pp90 = out["total_points_per90_10"] * out["minutes_mean_3"] / 90.0
    out["baseline_pp90_minutes"] = pp90.fillna(out["baseline_last5"]).fillna(0.0)
    return out


# --- Metrics -------------------------------------------------------------------------------

def _ndcg_at_k(y_true, y_score, k=20):
    """Normalised discounted cumulative gain, graded by actual points.

    Rewards putting genuine hauls near the top of the ranking, not merely ordering correctly:
    a 16-point return in your top 20 counts for more than a 6-point one.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    gain = np.clip(y_true, 0, None)
    order = np.argsort(-y_score, kind="stable")[:k]
    disc = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((gain[order] * disc).sum())
    ideal = np.sort(gain)[::-1][:k]
    idcg = float((ideal * 1.0 / np.log2(np.arange(2, len(ideal) + 2))).sum())
    return dcg / idcg if idcg > 0 else np.nan


def _precision_at_k(y_true, y_score, k):
    """Overlap between the predicted top-k and the actual top-k scorers."""
    if len(y_true) < k:
        return np.nan
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    pred_top = set(np.argsort(-y_score, kind="stable")[:k])
    true_top = set(np.argsort(-y_true, kind="stable")[:k])
    return len(pred_top & true_top) / k


def score_gameweek(y_true, y_pred, minutes, position_id=None, p_any=None):
    """All headline metrics for a single gameweek's predictions, as a {name: value} dict.

    Ranking metrics lead. MAE over all players is reported but must not be optimised: with 59%
    of rows at zero minutes it is minimised by predicting near-zero for everyone, which produces
    a useless model that looks excellent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    minutes = np.asarray(minutes, dtype=float)
    err = y_pred - y_true
    played = minutes > 0

    out = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "spearman_all": float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")),
        "ndcg_at_20": _ndcg_at_k(y_true, y_pred, 20),
        "n": float(len(y_true)),
    }
    for k in (10, 20, 50):
        out[f"precision_at_{k}"] = _precision_at_k(y_true, y_pred, k)

    if played.sum() > 1:
        out["mae_played"] = float(np.abs(err[played]).mean())
        out["rmse_played"] = float(np.sqrt((err[played] ** 2).mean()))
        # The headline. Restricting to players who actually featured strips out stage 1's easy
        # win (ranking non-players below players) and measures the hard part: ranking starters.
        out["spearman_likely"] = float(
            pd.Series(y_pred[played]).corr(pd.Series(y_true[played]), method="spearman")
        )

    # What the single highest-predicted player actually scored — the most intuitive number here,
    # and a brutally honest one.
    if len(y_true):
        out["captain_points"] = float(y_true[int(np.argmax(y_pred))])

    if p_any is not None:
        likely = np.asarray(p_any, dtype=float) >= 0.5
        if likely.sum() > 1:
            out["spearman_p_any_likely"] = float(
                pd.Series(y_pred[likely]).corr(pd.Series(y_true[likely]), method="spearman")
            )
    return out


# --- Two-stage model -----------------------------------------------------------------------
# FPL points are dominated by whether a player is on the pitch at all: 59% of rows have zero
# minutes. Predicting points directly makes one model serve two unrelated jobs — guessing team
# selection and guessing performance — and the zero spike drags every estimate toward zero.
#
# Stage 1 predicts playing time. Stage 2 predicts points EXCLUDING appearance points, fitted only
# on players who featured, so it never re-learns what stage 1 already encodes. They multiply.

FITTED_MODELS = ("linear_2stage", "gbm_2stage", "gbm_poisson_2stage", "blend_2stage")
BLEND_MEMBERS = ("linear_2stage", "gbm_2stage", "gbm_poisson_2stage")
ALL_MODELS = BASELINE_MODELS + FITTED_MODELS


PRIMARY_MODEL = "blend_2stage"

# Power BI slicer dimension. Keeping the baselines in the published table is deliberate: it lets
# the dashboard show what the model adds over simple arithmetic instead of asking for trust.
MODEL_DIM = pd.DataFrame([
    ("baseline_zero", "Baseline: zero", "baseline", 1, 0,
     "Predicts 0 for everyone. The floor — 59% of player-gameweeks really are zero."),
    ("baseline_season_mean", "Baseline: season average", "baseline", 1, 0,
     "Player's average points so far this season."),
    ("baseline_last5", "Baseline: last 5 average", "baseline", 1, 0,
     "Player's average points over their last five fixtures."),
    ("baseline_pp90_minutes", "Baseline: rate x minutes", "baseline", 1, 0,
     "Long-run points per 90 scaled by recent minutes — quality times availability, unfitted."),
    ("linear_2stage", "Ridge (two-stage)", "linear", 0, 0,
     "Logistic regression for playing time, ridge regression for points given playing."),
    ("gbm_2stage", "Gradient boosting (two-stage)", "gbm", 0, 0,
     "Histogram gradient boosting for both stages, squared-error loss."),
    ("gbm_poisson_2stage", "Gradient boosting, Poisson", "gbm", 0, 0,
     "As above with a Poisson loss, which suits count-like scoring."),
    ("blend_2stage", "Blend of all three models", "blend", 0, 1,
     "Average of the ridge, gradient boosting and Poisson predictions. Best on backtest."),
], columns=["model_name", "model_label", "model_family", "is_baseline", "is_primary",
            "description"])


def sample_weights(df):
    """Down-weight seasons played under the pre-2025-26 scoring rules.

    `defensive_contribution` became a scoring category in 2025-26, so a 2023-24 defender's
    total_points is drawn from a different points function. Full exclusion would cost ~80k rows;
    weighting keeps them while telling the model which era to trust.
    """
    return np.where(df["season"] < DC_RULE_SEASON, PRE_DC_SEASON_WEIGHT, 1.0)


def _build_estimator(kind, task):
    """One estimator. `kind` in {linear, gbm, gbm_poisson}; `task` in {clf, reg}."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if kind == "linear":
        # Ridge and LogisticRegression cannot take NaN, and the feature cliffs guarantee plenty
        # of it. add_indicator keeps "this was missing" as its own signal rather than pretending
        # the median was observed.
        pre = ColumnTransformer(
            transformers=[("pos", OneHotEncoder(handle_unknown="ignore"), ["position_id"])],
            remainder=Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
        )
        model = (LogisticRegression(max_iter=2000, C=1.0) if task == "clf"
                 else Ridge(alpha=5.0))
        return Pipeline([("pre", pre), ("model", model)])

    # HistGradientBoosting is a LightGBM-class histogram GBM shipped inside scikit-learn, and it
    # handles NaN natively — which matters because the feature availability cliffs are structural,
    # not random. Using it keeps the prediction job at two extra packages instead of five.
    common = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=40, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.1, random_state=0)
    if task == "clf":
        return HistGradientBoostingClassifier(**common)
    loss = "poisson" if kind == "gbm_poisson" else "squared_error"
    return HistGradientBoostingRegressor(loss=loss, **common)


def _fit(kind, task, X, y, w):
    """Fit one estimator, routing sample_weight through the Pipeline where there is one."""
    est = _build_estimator(kind, task)
    key = "model__sample_weight" if kind == "linear" else "sample_weight"
    return est.fit(X, y, **{key: w})


def fit_two_stage(train, features, kind):
    """Fit stage 1 (two classifiers) and stage 2 (one regressor) on `train`."""
    X, w = train[features], sample_weights(train)
    clf_any = _fit(kind, "clf", X, (train["minutes"] > 0).astype(int), w)
    clf_60 = _fit(kind, "clf", X, (train["minutes"] >= 60).astype(int), w)

    # Stage 2: points net of appearance points, fitted only on players who actually featured.
    played = train[train["minutes"] > 0]
    y_rest = (played["total_points"].to_numpy(dtype=float)
              - np.where(played["minutes"] >= 60, 2.0, 1.0))
    if kind == "gbm_poisson":
        y_rest = np.clip(y_rest, 0, None)  # Poisson deviance is undefined for negatives
    X2 = played[features].assign(is_60plus=(played["minutes"] >= 60).astype(int))
    reg = _fit(kind, "reg", X2, y_rest, sample_weights(played))

    return {"kind": kind, "clf_any": clf_any, "clf_60": clf_60, "reg": reg, "features": features}


def predict_two_stage(models, df):
    """Expected points, decomposed. Returns a frame aligned to `df.index`."""
    features = models["features"]
    X = df[features]

    p_any = models["clf_any"].predict_proba(X)[:, 1]
    p_60 = np.minimum(models["clf_60"].predict_proba(X)[:, 1], p_any)
    p_cameo = np.clip(p_any - p_60, 0, None)

    # Run stage 2 twice — once as if they start, once as if they come off the bench — so the
    # combination below is a genuine conditional expectation rather than one blurred average.
    ep_60 = models["reg"].predict(X.assign(is_60plus=1))
    ep_cameo = models["reg"].predict(X.assign(is_60plus=0))

    predicted = (2.0 * p_60 + 1.0 * p_cameo) + p_60 * ep_60 + p_cameo * ep_cameo
    return pd.DataFrame({
        "p_any": p_any,
        "p_60": p_60,
        "expected_minutes": 75.0 * p_60 + 25.0 * p_cameo,
        "ep_rest_60": ep_60,
        "ep_rest_cameo": ep_cameo,
        "predicted_points": predicted,
    }, index=df.index)

