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

An experimental FBref scraper also exists (`scripts/04_fpl_detailed_player_backup_scraper.py`) as a backup if other data degrades.

## Updates

- **Updated daily during the season**

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

**Dimension tables**

| File               | Contents                                     |
| ------------------ | -------------------------------------------- |
| `dim_player.csv`   | Persistent cross-season player IDs and names |
| `dim_team.csv`     | Persistent cross-season team IDs and names   |
| `dim_position.csv` | Position ID mapping (GK / DEF / MID / FWD)   |
| `dim_fixture.csv`  | Historical fixture list with persistent IDs  |
| `dim_season.csv`   | Season ID mapping                            |

`fact_player_next_fixtures.csv` is a **snapshot**, rebuilt in full on every run rather than
accumulated, because "the next five" moves as the season advances. It is in long format — five rows
per player, ordered by `fixture_order` — so the horizon isn't fixed by the schema and difficulty can
be aggregated. Opponent codes also carry the venue in their case (`BOU` home, `bou` away), with
`is_home` available as a proper boolean.

## Credits

- [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) — per-match player & team stats (olbauday)
- [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) — historical data (vaastav)
