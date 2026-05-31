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
- **FBref** — supplementary match stats
- **Fantasy-Premier-League archive** (by [vaastav](https://github.com/vaastav/Fantasy-Premier-League)) — historical seasons (2020–25)

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

## Credits

- [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) — per-match player & team stats (olbauday)
- [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) — historical data (vaastav)
