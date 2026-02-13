# Research Prompts for Basketball ML Betting Project

Run each of these as separate agent prompts with web search/fetch enabled.

---

## 1. Betting Exchanges Research

Research peer-to-peer basketball betting marketplaces/exchanges where you bet against other players rather than against a sportsbook. I need thorough, current information on:

1. **Major betting exchanges** - Betfair, Sporttrade, Prophet Exchange, Novig, Drift, and any others. For each:
   - How they work (order book style? parimutuel?)
   - Fee/commission structure
   - Whether they offer college basketball markets
   - US availability/legality (which states)
   - API access for programmatic betting
   - Liquidity levels
   - Minimum bets

2. **Crypto/decentralized options** - Polymarket, Azuro, SX Bet, Overtime Markets, and others
   - How they work
   - Liquidity for college basketball
   - Fees
   - Any regulatory concerns

3. **Why exchanges are better for sharp bettors** vs traditional sportsbooks (no getting limited, better odds, etc.)

4. **Comparison of exchange commission rates** vs sportsbook vig

---

## 2. Basketball ML Features & Modeling

Research machine learning approaches for predicting basketball game outcomes, specifically focused on finding profitable betting edges. Be thorough and specific. Cover:

1. **Key features/metrics for basketball prediction models**, organized by category:
   - Pace/tempo metrics (possessions per game, play count, etc.)
   - Offensive efficiency metrics (points per possession, eFG%, TS%, etc.)
   - Defensive efficiency metrics
   - Rebounding metrics (offensive/defensive rebounding rate)
   - Turnover metrics
   - Four Factors (Dean Oliver's framework)
   - Player-level metrics (PER, BPM, RAPTOR, etc.)
   - Lineup/rotation data
   - Rest days, travel, schedule density
   - Home court advantage quantification
   - Injury/roster availability

2. **The "one feature at a time" approach** - why it works, how to build up complexity gradually, starting with pace/play count

3. **Specific ML model types** that work well for basketball:
   - Linear/logistic regression as baseline
   - Gradient boosted trees (XGBoost, LightGBM)
   - Neural networks
   - Bayesian approaches
   - Elo/rating systems (how they compare)

4. **Feature engineering best practices**:
   - Rolling averages vs season averages
   - Opponent-adjusted metrics
   - Recency weighting
   - How to handle roster changes (especially in college)

5. **Common pitfalls**:
   - Overfitting
   - Lookahead bias
   - Survivorship bias
   - Sample size issues
   - Collinearity between features

6. **What the academic literature says** about basketball prediction and market efficiency

Be specific with metric definitions and formulas where possible.

---

## 3. Basketball Data Sources

Research all available data sources for college basketball (NCAA) and NBA data that could be used for building ML betting models. Be thorough and specific:

1. **Free data sources**:
   - Sports Reference / Basketball Reference (what's available, scraping policies)
   - ESPN API (endpoints, what data is available)
   - NCAA API/stats site
   - KenPom (what's free vs paid)
   - Barttorvik (what's available)
   - CBBAnalytics
   - Any other free sources

2. **Paid data sources**:
   - KenPom subscription (cost, what you get)
   - Synergy Sports
   - Sportradar
   - Stats Perform
   - Any others

3. **Python libraries/packages** for accessing basketball data:
   - `sportsipy` / `sportsreference`
   - `nba_api`
   - `ncaa_api` or equivalent
   - `hoopR` (R package, but worth mentioning)
   - Any others

4. **Historical betting odds data**:
   - Where to get historical spreads, totals, moneylines
   - Kaggle datasets
   - sportsbookreviewsonline.com
   - Other sources

5. **Play-by-play data availability** for college basketball specifically

6. **Data update frequency** - how quickly can you get box scores, play-by-play after games

7. **Specific API endpoints or scraping approaches** that are most reliable

Include URLs and pricing where available.

---

## 4. College Basketball Edge & Strategy

Research why college basketball specifically offers the best opportunities for profitable ML-based betting, and what strategies successful quantitative bettors use. Cover:

1. **Why college basketball markets are inefficient**:
   - Number of teams (350+ D1) vs analyst/bettor coverage
   - Information asymmetry in small conferences
   - Transfer portal impact on roster turnover
   - Early season vs late season line accuracy
   - Conference tournament and March Madness specifics

2. **Specific betting markets to target**:
   - Spreads vs totals vs moneylines
   - First half vs full game
   - Player props
   - Live/in-game betting
   - Which markets tend to be softest in college basketball

3. **Bankroll management**:
   - Kelly Criterion and fractional Kelly
   - Unit sizing
   - How to handle variance/drawdowns
   - Expected ROI ranges for successful models (be realistic)
   - How many bets per season a college basketball model might generate

4. **Backtesting methodology**:
   - Walk-forward validation
   - Out-of-sample testing
   - How to measure edge (CLV - Closing Line Value)
   - Statistical significance testing (how many bets before you know your edge is real)

5. **What successful basketball quants have shared publicly**:
   - Any public interviews, blog posts, or papers from people who've done this
   - Common strategies that are known to work
   - Haralabos Voulgaris approach (if documented)
   - Any academic papers on basketball market efficiency

6. **Legal considerations**:
   - US state-by-state legality overview
   - Tax implications of sports betting winnings
   - Whether using models/algorithms is legal everywhere

7. **Realistic expectations**:
   - What ROI% is achievable
   - How long until statistical significance
   - Capital requirements
   - Time investment required

Be honest and realistic — don't oversell the opportunity but do highlight where genuine edges exist.

---

## Instructions

For each prompt above, write the full research output to a separate file:
- `research/01-exchanges.md`
- `research/02-ml-features.md`
- `research/03-data-sources.md`
- `research/04-strategy.md`

Then compile the key findings into a single `research/00-summary.md`.
