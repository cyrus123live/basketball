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
- **Barttorvik** (barttorvik.com) — free KenPom alternative, AdjO/AdjD/AdjT/Four Factors, scrapeable, data back to 2008
- **ESPN API** — undocumented but reliable, box scores, PBP, schedules, some odds. Use `cbbpy` Python package.
- **SBRO** (sportsbookreviewsonline.com) — free historical closing lines (spreads, totals, ML) back to ~2007-08

### Cheap ($20-50/yr)
- **KenPom** ($20/yr) — gold standard for college basketball efficiency ratings. Use `kenpompy` Python package.
- **The Odds API** ($20/mo+) — real-time odds from multiple books, essential for CLV tracking

### Python Stack
- `cbbpy` — college basketball box scores and PBP from ESPN
- `kenpompy` — KenPom data (requires subscription)
- `nba_api` — NBA data from NBA.com (NBA only, not college)
- `sportsipy` — Sports Reference scraper (fragile, may need patching)
- `hoopR` (R) — most comprehensive college basketball package if willing to use R

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
├── CLAUDE.md              # This file
├── research-prompts.md    # Prompts for deep research agents
├── research/              # Research outputs (from Docker runs)
│   ├── 00-summary.md
│   ├── 01-exchanges.md
│   ├── 02-ml-features.md
│   ├── 03-data-sources.md
│   └── 04-strategy.md
├── data/                  # Raw and processed data
│   ├── raw/               # Downloaded/scraped data
│   ├── processed/         # Cleaned features
│   └── odds/              # Historical and live odds
├── models/                # Trained models and configs
├── notebooks/             # Exploration and analysis
├── src/                   # Source code
│   ├── data/              # Data collection and processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and evaluation
│   ├── betting/           # Bet sizing, execution, tracking
│   └── utils/             # Shared utilities
├── backtests/             # Backtest results and analysis
└── logs/                  # Bet logs, P&L tracking
```

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
