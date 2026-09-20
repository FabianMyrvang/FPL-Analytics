# Fantasy Premier League Data and Power BI report

This is a Fantasy Premier League Analytics project that collects and combines Fantasy Premier League data into clean, ready-to-use datasets and visualized in Power BI.

Pulls Premier League data from several open sources, cleans and combines it into consistent
per-player and per-match datasets with stable cross-season IDs, and refreshes automatically
during the season.

## Dashboard

An interactive Power BI dashboard, built on these datasets, is published live and refreshes
automatically as the data updates:

**▶ [View the live dashboard](https://app.powerbi.com/view?r=eyJrIjoiNDNlOGI4NzUtZTE3YS00NzdiLWJlNzktZmJkOGJjYWMwY2RmIiwidCI6IjVhZTVlNDFkLTM5OGQtNDk1NC1hOWQwLTU5YTdmNTVkZDU1NyJ9)**

## Sources:

- **FPL API** — player info, gameweek scores, prices, transfers, fixtures
- **FPL-Core-Insights** (by [olbauday](https://github.com/olbauday/FPL-Core-Insights)) — detailed per-match player & team stats
- **Fantasy-Premier-League archive** (by [vaastav](https://github.com/vaastav/Fantasy-Premier-League)) — historical seasons (2020–25), used to seed the first run

An experimental FBref scraper also exists (`scripts/04_fpl_elo_player_backup_scraper.py`) as a
fallback for when the FPL-Core-Insights data degrades mid-season. It is not part of the automated
pipeline, and none of the published datasets currently come from it.

## Updates

The datasets refresh themselves via a [GitHub Action](.github/workflows/update-data.yml):

- **Every 4 hours** during the season (GitHub's scheduler is best-effort and often runs late,
  so the pipeline polls rather than relying on one well-timed run)
- **Paused over the off-season** — June, July and the first 24 days of August — resuming
  automatically each 25 August, once the new season's opening gameweek is complete
- A run only commits when the data actually changed, so quiet periods produce no commits

The predictions refresh via [a second Action](.github/workflows/predict-points.yml), which runs
after the data pipeline and only when a gameweek has actually completed — so they update roughly
once per gameweek rather than on a clock. It retrains from scratch each time.

## Output

The datasets live in [`FPL_DATA/`](FPL_DATA/) as CSVs and can be used directly.

**Fact tables**

| File                            | Contents                                                                                   | Seasons           | Source                       |
| ------------------------------- | ------------------------------------------------------------------------------------------ | ----------------- | ---------------------------- |
| `fact_fpl_player_gw.csv`        | Per-player per-gameweek FPL scoring: points, goals, assists, xG, minutes, price, ownership | 2020–21 → present | FPL API + vaastav archive    |
| `fact_fpl_fixture.csv`          | Fixture results and FPL difficulty ratings                                                 | 2020–21 → present | FPL API + vaastav archive    |
| `fact_detailed_player_gw.csv`   | Per-player per-match detailed stats: shots, passes, duels, dribbles, etc.                  | 2025–26 → present | FPL-Core-Insights (olbauday) |
| `fact_detailed_fixture.csv`     | Per-match detailed team stats with ELO ratings                                             | 2025–26 → present | FPL-Core-Insights (olbauday) |
| `fact_player_next_fixtures.csv` | One row per player per upcoming fixture: opponent, venue, FPL difficulty (1–5)             | current season    | FPL API                      |
| `fact_player_points_prediction.csv` | Predicted points per player for the next 3 gameweeks, one row per model               | next 3 gameweeks  | model (see below)            |
| `fact_player_points_prediction_history.csv` | What was predicted before each gameweek was played              | 2026–27 → present | model                        |
| `fact_model_backtest_metric.csv` | Walk-forward accuracy of every model, per gameweek and metric                            | 2025–26 → present | model                        |

**Dimension tables**

| File               | Contents                                     |
| ------------------ | -------------------------------------------- |
| `dim_player.csv`   | Persistent cross-season player IDs and names |
| `dim_team.csv`     | Persistent cross-season team IDs and names   |
| `dim_position.csv` | Position ID mapping (GK / DEF / MID / FWD)   |
| `dim_fixture.csv`  | Historical fixture list with persistent IDs  |
| `dim_season.csv`   | Season ID mapping                            |
| `dim_model.csv`    | Prediction models and their descriptions     |

`fact_player_next_fixtures.csv` is a **snapshot**, rebuilt in full on every run rather than
accumulated, because "the next five" moves as the season advances. It is in long format — five rows
per player, ordered by `fixture_order` — so the horizon isn't fixed by the schema and difficulty can
be aggregated. Opponent codes also carry the venue in their case (`BOU` home, `bou` away), with
`is_home` available as a proper boolean.

## Points prediction

`fact_player_points_prediction.csv` holds predicted points for every player for the next three
gameweeks. Predictions come from a **two-stage model**: one stage estimates whether the player
will feature at all (and for 60+ minutes), a second estimates what they score given that they
play, and the two are combined into an expected value. FPL points are dominated by team
selection — 59% of player-gameweeks are zero minutes — so separating the two questions works
considerably better than predicting points directly.

Features are lagged rolling windows of past performance, fixture difficulty, venue, rest days,
opponent strength and price. Everything is shifted strictly backwards by the prediction horizon,
so a 3-gameweek-ahead prediction never sees results that would not yet have happened.

Eight models are published side by side, including four deliberately simple baselines, so you can
see what the model actually adds. Accuracy from walk-forward backtesting over 2025–26 (one
gameweek ahead):

| Model | Rank correlation¹ | Top-20 hit rate² | RMSE |
| --- | --- | --- | --- |
| Blend of all three (primary) | **0.36** | **0.18** | **1.93** |
| Best simple baseline | 0.30 | 0.14 | 2.12 |

¹ Spearman correlation among players who actually featured — how well the ordering matches.
² Share of the predicted top 20 that landed in the real top 20 for that gameweek.

**What this is and is not.** It predicts *ranking* far better than exact scores. Single-gameweek
points are close to irreducibly noisy — a 9-point haul often turns on one deflected shot — so
treat the output as an ordering of who is likely to do well, not a forecast. The playing-time
stage is the most reliable part (0.95 AUC, well calibrated).

**Known limitation:** injury and suspension news is not an input. The model infers availability
from minutes history alone, so it will confidently predict a full game for someone who picked up
a knock in training this week. Check the news before acting on it.

Full detail, including every metric per gameweek, is in `fact_model_backtest_metric.csv`.
A spreadsheet of the next gameweek's predictions is written to
[`exports/`](exports/) on each run.

## Credits

- [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) — per-match player & team stats (olbauday)
- [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) — historical data (vaastav)
