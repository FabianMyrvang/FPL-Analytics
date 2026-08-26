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

- **Daily at 06:00 UTC** during the season (GitHub's scheduler often runs it 30–60 minutes late)
- **Paused over the off-season** — June, July and the first 24 days of August — resuming
  automatically each 25 August, once the new season's opening gameweek is complete
- A run only commits when the data actually changed, so quiet periods produce no commits

## Output

The datasets live in [`FPL_DATA/`](FPL_DATA/) as CSVs and can be used directly.

**Fact tables**

| File                    | Contents                                                                        | Seasons           | Source |
| ----------------------- | ------------------------------------------------------------------------------- | ----------------- | ------ |
| `fpl_gameweek_fact.csv` | Per-player per-gameweek FPL scoring: points, goals, assists, xG, minutes, price | 2020–21 → present | FPL API + vaastav archive |
| `fpl_fixture_fact.csv`  | Fixture results and FPL difficulty ratings                                      | 2020–21 → present | FPL API + vaastav archive |
| `elo_gameweek_fact.csv` | Per-player per-match detailed stats: shots, passes, duels, dribbles, etc.       | 2025–26 → present | FPL-Core-Insights (olbauday) |
| `elo_fixture_fact.csv`  | Per-match team stats with ELO ratings                                           | 2025–26 → present | FPL-Core-Insights (olbauday) |

**Dimension tables**

| File               | Contents                                     |
| ------------------ | -------------------------------------------- |
| `player_dim.csv`   | Persistent cross-season player IDs and names |
| `team_dim.csv`     | Persistent cross-season team IDs and names   |
| `position_dim.csv` | Position ID mapping (GK / DEF / MID / FWD)   |
| `fixture_dim.csv`  | Historical fixture list with persistent IDs  |
| `season_dim.csv`   | Season ID mapping                            |

**Helper tables**

| File                        | Contents                                                                     |
| --------------------------- | ---------------------------------------------------------------------------- |
| `player_next_fixtures.csv`  | Each current-season player's next five fixtures: opponent short code and FPL difficulty (1–5) |

`player_next_fixtures.csv` is a **snapshot**, rebuilt in full on every run rather than accumulated,
because "the next five" moves as the season advances. Opponent codes use case to show venue —
`BOU` is home, `bou` is away.

## Credits

- [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) — per-match player & team stats (olbauday)
- [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) — historical data (vaastav)
