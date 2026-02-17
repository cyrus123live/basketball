# Basketball ML Betting Project

## Goal
Build a machine learning system to find profitable edges in basketball betting markets, primarily college basketball (NCAA). The end goal is to make money by betting against other players on peer-to-peer exchanges rather than against sportsbooks.

## Why Basketball / Why College
- Basketball has the highest correlation between individual player ability and game outcome of any major sport — less luck, more signal
- High game frequency (~5,000+ D1 games/season) provides large sample sizes for training and betting volume
- College basketball (363 D1 teams, 32 conferences) is structurally inefficient:
  - Most analyst/sharp coverage focuses on ~60 Power conference teams, leaving 300+ under-covered
  - Transfer portal causes massive year-to-year roster turnover, creating information gaps
  - Early season lines are significantly less accurate than late season
  - Small conference games are set largely by algorithm with minimal human oversight
  - Totals and first-half markets are softer than spreads

## Core Approach: One Feature at a Time
Advice from someone who does this professionally. Don't throw everything into a model — build up complexity incrementally:

1. **Start with pace/play count** — establishes tempo context, affects variance, is the denominator for all efficiency metrics
2. **Add net rating (ORtg - DRtg)** — single strongest predictor of future performance
3. **Add Four Factors one by one** — eFG% → TOV% → ORB% → FTr (Dean Oliver's framework, in order of importance: 40/25/20/15%)
4. **Add contextual features** — rest days, home court advantage, travel
5. **Add opponent adjustments** — strength of schedule (KenPom-style iterative adjustment)
6. **Add player availability** — injury impact, roster changes
7. **Add recency weighting** — exponential decay to capture team trajectory

At each step, validate on held-out data. Stop adding features when marginal improvement plateaus.

## Key Metrics & Formulas
```
Possessions = FGA - OREB + TOV + 0.44 * FTA  (use 0.40 for college)
Pace = Poss / (Minutes / 5)
ORtg = (Points / Poss) * 100
eFG% = (FGM + 0.5 * 3PM) / FGA
TS% = Points / (2 * (FGA + 0.44 * FTA))
TOV% = TOV / (FGA + 0.44 * FTA + TOV)
ORB% = OREB / (OREB + Opp_DREB)
FTr = FTA / FGA
```

## Betting Strategy
- **Target markets (soft → sharp):** player props > first half totals > full game totals > first half spreads > full game spreads (low-profile) > full game spreads (high-profile)
- **Best edge windows:** early season (Nov-Dec), small conference games, games with heavy roster turnover, late-breaking injury situations
- **Bankroll management:** fractional Kelly (1/4 to 1/2 Kelly), never full Kelly. 1 unit = 1-2% of bankroll.
- **Measure edge with CLV** (Closing Line Value), not just win rate. CLV is less noisy and more predictive of long-term profitability.
- **Realistic ROI:** 1-4% for a good individual model, 3-7% for elite. Break-even at -110 requires 52.4% ATS. Anyone claiming 10%+ on volume is lying.
- **Statistical significance:** Need 600+ bets at 55% observed win rate to confirm edge at 95% confidence. Plan for 2-3 seasons minimum.

## Where to Bet: Peer-to-Peer Exchanges
The user specifically wants to bet against other players, not books. Key advantages: no account limiting, lower fees, ability to set your own lines.

### US-Legal Options
- **Sporttrade** (NJ, CO+) — financial exchange model, ~2% commission, developing API, college basketball available
- **Novig** (NJ, IN+) — P2P matching, ~1% embedded vig, consumer UX (no API), college basketball available
- **Prophet Exchange** (NJ) — order book, ~2-3% commission, developing API
- **Kalshi** (CFTC-regulated, most US) — prediction market, expanding into sports, worth monitoring

### Crypto/Decentralized (unregulated, use at own risk)
- **SX Bet** — blockchain order book, 2-4% commission, has API, some NCAAB
- **Overtime Markets** — AMM on Optimism, 2-3% spread, some NCAAB
- **Azuro** — DeFi betting protocol, 3-5% spread, limited NCAAB

### Non-US
- **Betfair Exchange** — gold standard, mature API, 2-5% commission, not available in US

## Data Sources

### Free (Priority)
- **ESPN JSON API** — undocumented but reliable JSON endpoints for scoreboard + boxscores. We use these directly (faster than cbbpy HTML scraping). Scoreboard: 1 request/day for all games. Summary: 1 request/game for full boxscore.
- **Barttorvik** (barttorvik.com) — free KenPom alternative, AdjO/AdjD/AdjT/Four Factors, scrapeable, data back to 2008. Requires JS verification workaround (POST `js_test_submitted=1`).
- **SBRO** (sportsbookreviewsonline.com) — free historical closing lines (spreads, totals, ML) back to ~2007-08

### Cheap ($20-50/yr)
- **KenPom** ($20/yr) — gold standard for college basketball efficiency ratings. Use `kenpompy` Python package.
- **The Odds API** ($20/mo+) — real-time odds from multiple books, essential for CLV tracking

### Python Stack
- `requests` + ESPN JSON API — primary data source for box scores (replaced cbbpy HTML scraping for ~8x speedup)
- `pandas` + `pd.read_html` — Barttorvik scraping (with `StringIO` workaround)
- `kenpompy` — KenPom data (requires subscription, not yet integrated)
- `thefuzz` — fuzzy team name matching for SBRO odds → game matching

## Model Architecture

### Start Simple
1. **Baseline:** Logistic regression with Four Factors differentials — remarkably hard to beat
2. **Level up:** XGBoost/LightGBM with ~15-20 features (shallow trees, max_depth 3-5, aggressive regularization)
3. **Advanced:** Bayesian hierarchical model (partial pooling across conferences, uncertainty estimates for Kelly sizing)
4. **Consider:** Team embeddings for college basketball's 363-team space

### Validation
- **Walk-forward only.** Train on seasons 1-N, validate on season N+1. Never k-fold for temporal data.
- **No lookahead bias.** Every prediction must use only data available before tip-off.
- **Compare against closing line** — if your model can't beat the closing line, it has no edge.

### Feature Engineering
- Use multiple rolling windows (5, 10, 15, 20 games) as separate features
- Exponential decay weighting (alpha 0.90-0.97) instead of hard rolling windows
- Opponent-adjust iteratively (KenPom-style: adjust offense for opponent defense quality, iterate until convergence)
- For college roster changes: blend preseason projections with in-season data using `w = games / (games + k)` where k ≈ 10-15
- Watch out for collinearity — Four Factors are designed to minimize it, but don't also add ORtg (it's derived from them)

## Common Pitfalls to Avoid
- **Overfitting:** rule of thumb — no more than 1 feature per 10-20 training observations
- **Lookahead bias:** the #1 methodological error. Never use season averages to predict mid-season games.
- **Survivorship bias:** testing 50 strategies and reporting the best one guarantees a fake edge
- **Opponent 3P% is mostly noise:** stabilizes after 50+ games, don't weight it heavily
- **Defensive metrics stabilize slower than offensive** (~25-30 games vs 15-20) — regress defense toward mean more aggressively early season
- **KenPom numbers are already priced in** — the market uses them. Your edge must come from what KenPom doesn't capture (injuries, travel, motivation, matchup-specific factors, speed of updates)

## Project Structure
```
basketball/
├── CLAUDE.md                        # This file
├── README.md                        # Project overview and quick start
├── PLAN.md                          # Multi-phase implementation plan
├── pyproject.toml                   # Dependencies (pip install -e .)
├── .env                             # API keys (never committed)
├── .gitignore
├── research-prompts.md              # Prompts for deep research agents
├── research/                        # Research outputs
├── src/
│   ├── data/
│   │   ├── cbbpy_collector.py       # ESPN JSON API box scores (~0.7s/game)
│   │   ├── barttorvik_collector.py  # Barttorvik team ratings scraper
│   │   ├── odds_api_collector.py    # The Odds API client
│   │   └── sbro_loader.py          # SBRO historical odds parser
│   ├── features/
│   │   └── compute.py              # Four Factors, pace, EWMA features
│   ├── models/                     # Model training (Phase 1)
│   ├── betting/                    # Bet sizing (Phase 2)
│   └── utils/
│       ├── db.py                   # SQLite FeatureStore (anti-lookahead)
│       └── config.py               # Paths, constants, helpers
├── scripts/
│   ├── build_historical.py         # One-time dataset assembly (~45 min/season)
│   ├── daily_collect.py            # Daily cron pipeline
│   └── paper_trade.py              # Naive LogReg paper trading
├── tests/                          # 56 unit/integration tests
├── data/
│   ├── basketball.db               # SQLite database (gitignored)
│   ├── raw/                        # CSV checkpoints (gitignored)
│   ├── odds/sbro/                  # SBRO Excel files (manual download)
│   └── team_name_crosswalk.csv     # ESPN/Barttorvik/SBRO name mapping
├── notebooks/                      # Exploration and analysis
├── models/                         # Trained models
├── backtests/                      # Backtest results
└── logs/                           # Daily pipeline logs
```

## Phase 0 Status (Complete)
- Data collectors: ESPN JSON API (box scores), Barttorvik (ratings), Odds API (live lines), SBRO (historical odds)
- SQLite feature store with 7 tables, anti-lookahead temporal queries
- Feature engine: possessions, pace, ORtg/DRtg, Four Factors via EWMA
- Naive paper trading pipeline (logistic regression, 2 features)
- 56 tests passing
- **Historical dataset built:** 63,130 games across 2016-2026, features computed, quality checks passed (10 box score mismatches out of 63K = 0.016%)
- **Daily cron pipeline:** `scripts/daily_cron.sh` daemon + `scripts/daily_collect.py` with absolute log paths. Tested: collects games, Barttorvik ratings, and Odds API lines in ~104s. Set up via crontab on a persistent machine: `0 11 * * * .venv/bin/python scripts/daily_collect.py`
- **SBRO odds loaded:** 14 files (2008-2021), 8,007 records matched to games across 2016-2021 (33.5% match rate — SBRO covers DonBest rotation only, not all D1 games). ~1,300 games/season with closing spreads and totals. 2022-2025 closing lines TBD (Odds API historical or scraping).
- **SBRO name matching:** `sbro_loader.py` uses `_normalize_name()` to bridge ESPN full names (`Miami (OH) RedHawks`) and SBRO compressed names (`MiamiOhio`) — camelCase splitting, mascot removal, abbreviation expansion, exact substitutions. Threshold: fuzzy score >= 75.

## Phase 1 Status (Complete — Baseline KILL)
- **SBRO swap bug fixed:** ~28% of records had total_close/spread_close swapped. Heuristic: if `|total| < |spread|`, swap. Also handled at query time via SQL CASE.
- **Walk-forward validation framework:** `src/models/walk_forward.py` — trains on prior seasons, validates on next. Folds: train 2016-18→val 2019, train 2016-19→val 2020, train 2016-20→val 2021.
- **Evaluation framework:** `src/models/evaluation.py` — regression metrics, CLV metrics, simulated betting at -110 juice, calibration by bucket, binomial significance test. 20 unit tests passing.
- **Feature build-up:** `src/models/totals_baseline.py` — adds features one at a time (pace → ortg → drtg → eFG → TOV% → ORB% → FTr → rest → min_games). Combined/averaged features for totals (not differentials).
- **Backtest script:** `scripts/backtest_totals.py` — runs full pipeline, prints results table, saves JSON to `backtests/`.
- **Results (LinearRegression, 13 features, ~4,850 eval games across 3 seasons):**
  - Model MAE: 14.00 vs closing line MAE: 13.01 — market is ~1 point more accurate
  - Directional accuracy: 50.3% (below 52.4% break-even)
  - Simulated ROI: -3.5% (losing money at -110 juice)
  - Not statistically significant (p = 0.99)
  - Ridge regression marginally better but same conclusion
- **Verdict: KILL** — raw EWMA features via linear regression cannot beat the closing line on totals. The market already prices in pace/efficiency/Four Factors.
- **Next:** The infrastructure is solid. To find edge, need features the market doesn't capture: opponent-adjusted ratings (KenPom-style), Barttorvik integration, situational factors (travel, rest, rivalry), or non-linear models (XGBoost) to find interaction effects.

## Key References
- Dean Oliver, "Basketball on Paper" (2004) — Four Factors framework
- KenPom.com / Barttorvik.com — adjusted efficiency methodology
- Nate Silver, "The Signal and the Noise" — sports betting chapter
- Joe Peta, "Trading Bases" — model-building methodology (baseball, but translates)
- Ed Miller & Matthew Davidow — "What Makes a Good Bet?" framework
- Paul & Weinbach (2005, 2007) — academic papers on college basketball market inefficiency
- Lopez & Matthews (2015) — Bayesian NCAA prediction models

## Legal Notes
- Using ML models for betting is legal everywhere sports betting is legal
- All winnings are taxable (federal + state). Keep detailed records of every bet.
- Books CAN and WILL limit sharp bettors — this is why we prefer exchanges
- Verify current state legality before placing any bets
