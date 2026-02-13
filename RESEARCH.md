# College Basketball ML Betting: Comprehensive Research

> Compiled February 2026. Data sources and platform details should be verified for current accuracy.

---

## Table of Contents

1. [Why Basketball (and College Basketball Specifically)](#1-why-basketball)
2. [Betting Exchanges & Peer-to-Peer Platforms](#2-betting-exchanges)
3. [ML Features & Modeling Approaches](#3-ml-features--modeling)
4. [Data Sources & Python Libraries](#4-data-sources)
5. [Strategy & Bankroll Management](#5-strategy--bankroll)
6. [Backtesting Methodology](#6-backtesting)
7. [Realistic Expectations](#7-realistic-expectations)
8. [Legal Considerations](#8-legal)
9. [References & Further Reading](#9-references)

---

## 1. Why Basketball

### Why Basketball Is the Best Sport for ML Betting

- **High game volume**: 30+ games per team per season (college), 82 (NBA). Compare to NFL's 17.
- **Low variance / high skill**: Basketball has the highest game-to-game predictability of major sports. The best team wins far more often than in baseball or football.
- **Individual impact**: Only 5 players on court; stars play 30-40 minutes. A single dominant player can swing a college team's entire profile. Individual talent translates to outcomes more directly than in football (22 starters, specialized units).
- **Pace normalization**: Possessions create a natural unit of analysis. Efficiency metrics (points per possession) are well-understood and predictive.

### Why College Basketball Markets Are Inefficient

**The scale problem:**
- 363 Division I teams across 32 conferences producing ~5,000+ games per season
- Sportsbooks cannot devote the same analytical resources per game as they can for 30 NBA teams
- Power conferences (SEC, Big Ten, Big 12, ACC, Big East) get the vast majority of coverage
- The remaining ~20+ conferences are dramatically under-covered

**Information asymmetry in small conferences:**
- A Tuesday night game between two mid-major teams may have lines set almost entirely by algorithm
- Local beat reporters, team insiders, and people who closely follow specific mid-major conferences have meaningfully better information than the market
- Injuries, suspensions, lineup changes, and coaching adjustments are often not priced in for these games

**Transfer portal impact:**
- Since NCAA relaxed transfer rules (one-time free transfer without sitting out), 1,500+ D1 players enter the portal annually
- Preseason projections are inherently less reliable
- Team chemistry takes time to develop, creating early-season volatility
- Historical team-level stats become less predictive; player-level tracking is essential
- Models that project transferred players in new systems have an edge

**Early season vs. late season:**
- November/December: books work with limited info (preseason rankings, recruiting ratings, returning players)
- Lines can be significantly off, especially for teams with heavy roster turnover
- By February/March, the market has "learned" from hundreds of games and lines sharpen
- **Your model's edge is likely largest in the first 6-8 weeks**, particularly for non-marquee matchups

**Conference tournaments and March Madness:**
- Small conference tournaments can be inefficient (autobiography to NCAA tournament at stake, thin coverage)
- March Madness itself is paradoxically one of the *most efficient* markets (massive public interest attracts sharp money)
- Niches exist: 12-vs-5 seed games, second-weekend fatigue/matchup specifics

**Quantifying the inefficiency:**
- Closing lines in college basketball are less efficient than NBA closing lines by roughly 1-3 points of accuracy on average
- Early-season lines show higher variance from true probabilities
- Totals tend to be slightly less efficient than spreads

---

## 2. Betting Exchanges

### Why Exchanges Over Sportsbooks

| Advantage | Detail |
|-----------|--------|
| **No account limiting** | Exchanges don't take the other side of your bet; they profit from commission regardless of who wins. Winning bettors are *valuable* to exchanges. |
| **Lower effective fees** | Traditional books: ~4.5-10% vig. Exchanges: 1-5% commission on net winnings only. |
| **Set your own odds** | Post orders at your desired price and wait for matching. |
| **Lay betting** | Bet against outcomes (act as the bookmaker). Doubles available strategies. |
| **Trading/position management** | Sell positions before settlement. Lock in profit, cut losses, arbitrage. |
| **Transparency** | See full order book depth, volume, price levels. |

### Effective Cost Comparison

| Platform | Fee Type | Typical Rate | Cost per $100 Wagered (50% WR) |
|----------|----------|-------------|-------------------------------|
| Traditional Book (-110/-110) | Vig | ~4.55% | ~$2.27 |
| Betfair Exchange | Commission on net winnings | 2-5% | $0.50-$1.25 |
| Sporttrade | Commission on net winnings | ~2% | ~$0.50 |
| Prophet Exchange | Commission on net winnings | 2-3% | $0.50-$0.75 |
| Novig | Embedded low vig | ~1% | ~$0.50 |
| SX Bet | Commission on net winnings | 2-4% | $0.50-$1.00 |
| Overtime Markets | AMM spread | 2-3% | $1.00-$1.50 |

### Major Regulated Exchanges (US)

#### Sporttrade
- **Model**: Continuous limit order book (financial exchange style). Outcomes priced $0-$1.
- **States**: New Jersey, Colorado (expanding)
- **Commission**: ~2% on net winnings
- **College basketball**: Yes (spreads, moneylines, totals)
- **API**: Beta/developing
- **Min bet**: ~$1-$2
- **Key feature**: Can sell positions before settlement (in-play trading)

#### Novig
- **Model**: Automated P2P matching engine. Feels like a sportsbook but is peer-to-peer behind the scenes.
- **States**: New Jersey, Indiana (expanding)
- **Commission**: ~1% embedded in odds ("near vig-free")
- **College basketball**: Yes
- **API**: No (consumer UX focused)
- **Min bet**: ~$1-$5
- **Key feature**: Lowest effective vig in the US market

#### Prophet Exchange
- **Model**: P2P order book (Betfair-style)
- **States**: New Jersey
- **Commission**: 2-3% on net winnings
- **College basketball**: Major conferences and tournaments
- **API**: Developing
- **Min bet**: ~$2-$5

#### Kalshi (CFTC-regulated prediction market)
- **Model**: Order book for event contracts
- **States**: Most US states (CFTC-regulated, not state gaming)
- **Expanding into sports**: Monitor for basketball markets
- **Key feature**: Most legally clear option for US-based prediction market trading

### International Exchanges (Not US)

#### Betfair Exchange (Gold Standard)
- World's largest exchange. Continuous double-auction order book.
- Commission: 5% on net winnings (2% for high volume)
- **Mature API**: Betfair API-NG. Full programmatic access. Widely used by algo bettors.
- NBA coverage excellent, NCAAB limited to major games/tournament
- NOT available to US residents

#### Smarkets
- UK-based. Clean UX, 2% commission. Robust API.
- Some basketball markets. Not available in US.

### Crypto / Decentralized Options

| Platform | Model | Basketball | Fees | US Legal? | API |
|----------|-------|-----------|------|-----------|-----|
| **SX Bet** | On-chain order book | NBA + some NCAAB | 2-4% net winnings | Unregulated | Yes |
| **Overtime Markets** | AMM (Optimism L2) | NBA + some NCAAB | 2-3% AMM spread | Unregulated | On-chain |
| **Polymarket** | Prediction market (Polygon) | Minimal/major events only | ~0% | Blocked for US | On-chain |
| **Azuro** | AMM/LP pool protocol | NBA, limited NCAAB | 3-5% spread | Unregulated | On-chain |
| **Drift BET** | Solana order book | Very limited | 0.05-0.1% | Unregulated | Solana SDK |

**Key takeaway for college basketball**: Exchange coverage for NCAAB is significantly thinner than NBA. Major games and tournaments have reasonable liquidity; regular-season mid-major games are often unavailable. You may need to use a combination of exchanges + traditional books.

---

## 3. ML Features & Modeling

### The "One Feature at a Time" Approach

Your contact's advice is sound. Starting with a single feature and building up forces you to:

1. **Understand causal structure** before adding complexity
2. **Avoid collinearity traps** (many basketball metrics are highly correlated)
3. **Prevent overfitting** through parsimony (each additional feature = additional degree of freedom to fit noise)
4. **Build intuition** for what drives outcomes and when the model behaves pathologically

### Key Features by Category

#### Pace / Tempo (Start Here)
- **Possessions per game**: `Poss = FGA - ORB + TOV + 0.475 * FTA`
- **Adjusted Tempo (AdjT)**: Pace adjusted for opponent's tempo
- Pace is excellent as a starting feature because:
  - It's a process metric (how teams play) rather than outcome metric (results)
  - More possessions = outcomes regress more toward mean (law of large numbers within a single game)
  - It's the denominator for all efficiency metrics
  - Pace differential reveals matchup dynamics

#### Offensive Efficiency
- **Offensive Rating (ORtg)**: Points scored per 100 possessions
- **Effective FG% (eFG%)**: `(FGM + 0.5 * 3PM) / FGA` — accounts for 3-pointers being worth more
- **True Shooting% (TS%)**: `PTS / (2 * (FGA + 0.44 * FTA))` — includes free throws
- **Points Per Possession (PPP)**: Raw efficiency measure

#### Defensive Efficiency
- **Defensive Rating (DRtg)**: Points allowed per 100 possessions
- **Opponent eFG%**: What opponents shoot against you
- **Opponent TOV%**: How often you force turnovers
- **Block% and Steal%**: Defensive activity metrics

#### Four Factors (Dean Oliver's Framework)
The Four Factors explain ~90% of point differential variance:

| Factor | Formula | Weight (Oliver) |
|--------|---------|----------------|
| **Shooting (eFG%)** | `(FGM + 0.5*3PM) / FGA` | 40% |
| **Turnovers (TOV%)** | `TOV / (FGA + 0.44*FTA + TOV)` | 25% |
| **Rebounding (ORB%)** | `ORB / (ORB + Opp_DRB)` | 20% |
| **Free Throws (FTr)** | `FTA / FGA` (free throw rate) | 15% |

Compute for both offense and defense = 8 features. Designed to minimize redundancy.

#### Rebounding
- **ORB%**: `ORB / (ORB + Opp_DRB)` — offensive rebounding rate
- **DRB%**: `DRB / (DRB + Opp_ORB)` — defensive rebounding rate
- **Total Rebound%**: Overall rebounding dominance

#### Player-Level Metrics
- **BPM (Box Plus/Minus)**: Estimates player's contribution in points per 100 possessions above average
- **PER (Player Efficiency Rating)**: All-in-one per-minute production metric
- **RAPTOR** (FiveThirtyEight): Hybrid box score + on/off metric
- **EPM (Estimated Plus-Minus)**: Regularized adjusted plus-minus
- **For college**: EvanMiya's BPR (Bayesian Performance Rating)

#### Situational Features
- **Rest days**: Days since last game (0 = back-to-back)
- **Travel distance**: Miles traveled to away game
- **Schedule density**: Games in last 7/14 days
- **Home court advantage**: Worth ~3-4 points in college (higher than NBA's ~2-3)
- **Altitude**: Relevant for games in places like BYU, Colorado, Wyoming

### Suggested Feature Build-Up Sequence

1. **Pace** — establishes tempo context
2. **Net Rating (ORtg - DRtg)** — single strongest predictor of future performance
3. **eFG% differential** — most important Four Factor
4. **TOV% differential** — second most important
5. **ORB% differential** — third
6. **FTr differential** — fourth
7. **Rest days differential** — schedule context
8. **Home court advantage** — location context
9. **Opponent-adjusted versions** of the above (strength of schedule)
10. **Player availability adjustments** — injury/roster context
11. **Recency-weighted versions** — captures team trajectory

At each step, validate on held-out data. Stop adding features when marginal improvement plateaus.

### ML Model Types

#### Logistic Regression (Baseline — Start Here)
```
P(win) = sigmoid(b0 + b1*X1 + b2*X2 + ... + bn*Xn)
```
- Highly interpretable, resistant to overfitting
- A well-tuned logistic regression with 10-15 good features is remarkably hard to beat
- Should be your first model and ongoing benchmark

#### Gradient Boosted Trees (XGBoost / LightGBM)
- Workhorse for structured/tabular data
- Naturally captures nonlinear relationships and interactions
- Key hyperparameters: `max_depth` (3-5), `learning_rate` (0.01-0.1), `n_estimators` (use early stopping)
- Feature importance output helps understand predictions
- Require more careful validation than linear models

#### Bayesian Approaches
- Prior distributions provide natural regularization
- Posterior distributions give uncertainty estimates (crucial for bet sizing)
- Hierarchical models useful for college basketball (conferences → teams)
- More computationally expensive (MCMC via Stan/PyMC)

#### Elo / Rating Systems
- Simple, transparent, handles time-series naturally
- Collapses everything to one number (can't capture multi-dimensional team profiles)
- Best used as a *feature* in a broader ML model
- FiveThirtyEight Elo: `New = Old + K * (Actual - Expected)`, with MOV adjustment

#### Neural Networks
- Rarely outperform GBMs on structured tabular data with <50 features
- Potentially useful for: very large datasets, heterogeneous data (play-by-play sequences), learning team embeddings
- Overkill for most basketball prediction tasks

### Feature Engineering Best Practices

#### Rolling Averages vs Season Averages
- Season averages weight a 3-month-old game the same as last night's — bad
- Use multiple rolling windows (5, 10, 15, 20 games) as separate features
- Metric stabilization times:
  - Pace: 5-8 games
  - TOV%: 10-15 games
  - eFG%: 15-20+ games
  - Opponent 3P%: 30+ games (mostly noise)

#### Exponential Decay Weighting
```
Weighted_metric = Sum(metric_i * alpha^(games_ago_i)) / Sum(alpha^(games_ago_i))
```
Alpha between 0.90-0.97. Avoids the "cliff" of hard rolling windows.

#### Opponent Adjustment (KenPom Method)
Raw metrics don't account for schedule strength. Iterative adjustment:
1. Start with raw ratings
2. Adjust each team's offense based on opponents' defensive ratings
3. Adjust each team's defense based on opponents' offensive ratings
4. Repeat until convergence (10-20 iterations)

#### Handling Roster Changes (College)
```
Preseason_estimate = w1*(Returning_production) + w2*(Recruit_talent) +
                     w3*(Transfer_impact) + w4*(Prior_year_rating * regression)

Current_estimate = (1 - w_season)*Preseason + w_season*In_season_adjusted
Where w_season = games_played / (games_played + k), k ≈ 10-15
```

### Common Pitfalls

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Overfitting** | Model fits noise in training data | Time-series CV, regularization, keep features low (~1 per 10-20 observations) |
| **Lookahead bias** | Future info leaks into features | Use ONLY data available before tip-off for each prediction |
| **Survivorship bias** | Selecting features that looked good historically by chance | Select features on domain knowledge, use holdout for evaluation |
| **Sample size** | 30-35 college games per season is tiny | Wide uncertainty bands early season, rely on priors |
| **Collinearity** | ORtg and eFG% correlate >0.80 | Use Four Factors (designed to minimize redundancy), check VIF, use regularization |

### Feature Importance for Betting (What Matters Most)

| Priority | Feature | Predictive Power | Market Pricing | Betting Edge |
|----------|---------|-----------------|----------------|-------------|
| 1 | Net Rating (opponent-adjusted) | Very High | Well-priced | Low |
| 2 | Four Factors | High | Well-priced | Low |
| 3 | Pace/Tempo | Moderate | Moderate | Moderate |
| 4 | Home Court | Moderate | Well-priced | Low |
| 5 | Rest/Schedule | Moderate | Increasingly well-priced | Low-Moderate |
| **6** | **Injury/Roster changes** | **High (situational)** | **Often mispriced (late news)** | **High** |
| **7** | **Lineup data** | **Low (noisy)** | **Poorly priced (ignored)** | **Moderate** |

**The key insight**: A feature with high predictive power but excellent market pricing gives you no edge. Features with moderate predictive power but poor market pricing (the market ignores or misprices them) are where money is made.

---

## 4. Data Sources

### Free Data Sources

#### Barttorvik (barttorvik.com) — RECOMMENDED PRIMARY
- **All free**. Team ratings, adjusted efficiency, Four Factors, game-by-game data, player stats
- T-Rank system, historical data back to 2008
- Transfer portal tracking
- Scrapeable with `requests` + `pandas.read_html()`
- More permissive than Basketball Reference

#### ESPN API (Undocumented but reliable)
- Scoreboard, box scores, play-by-play, teams, standings, rankings
- Near real-time during games
- Betting lines sometimes included in scoreboard data
- No advanced stats, no official API key system
- `cbbpy` Python package wraps this

```python
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
# Scoreboard: {BASE}/scoreboard?dates=YYYYMMDD&limit=200&groups=50
# Game summary: {BASE}/summary?event={gameId}
# Teams: {BASE}/teams?limit=400
```

#### Sports Reference / Basketball Reference
- Comprehensive stats, box scores, historical data
- **Strict rate limit**: 20 requests/minute. Will block aggressive scraping.
- Advanced stats, game logs, season stats
- College: sports-reference.com/cbb/
- NBA: basketball-reference.com

#### NCAA Stats (stats.ncaa.org)
- Official NCAA statistics for all divisions
- Historical data back to ~2009-10
- Slow, clunky, no API — must scrape HTML

#### Kaggle — March Machine Learning Mania
- Annual competition with extensive historical data
- Game results back to 2003, seeds, team ratings, some odds data
- Great for initial model development and backtesting

#### Historical Betting Odds — SBRO
- sportsbookreviewsonline.com
- **Free** downloadable Excel/CSV files
- Historical closing lines (spread, O/U, moneyline) back to ~2007-08
- Typically Pinnacle closing lines (sharpest market)
- Only closing lines — no opening lines or line movement

### Paid Data Sources (Worth It)

#### KenPom ($19.99/year) — STRONGLY RECOMMENDED
- Gold standard for college basketball adjusted efficiency
- AdjO, AdjD, AdjT, Four Factors, game-by-game predictions
- Historical data back to 2002
- Access via `kenpompy` Python package
- **Extremely high value** for the price

#### The Odds API (~$20/month)
- Real-time odds from multiple sportsbooks
- Historical odds on paid tiers
- College basketball + NBA coverage
- Clean REST API

#### EvanMiya ($9.99/season)
- Player-level Bayesian Performance Ratings
- Very good individual player advanced metrics for college

#### SportsDataIO
- NCAA basketball API
- Free trial: 1,000 calls/month
- Paid: starts ~$25-50/month
- Includes scores, stats, schedules, odds, some play-by-play

### Python Libraries

| Package | Covers | Data Source | Status |
|---------|--------|-------------|--------|
| `nba_api` | NBA only | NBA.com/stats | Active, mature |
| `cbbpy` | College basketball | ESPN API | Active |
| `kenpompy` | College basketball | KenPom (requires $20 sub) | Active |
| `sportsipy` | NBA + College | Sports Reference | Inconsistent maintenance |
| `hoopR` (R) | College + NBA | ESPN, NBA.com, KenPom | Very active (SportsDataverse) |

### Recommended Data Stack

#### Budget: $0 (Free Only)
| Need | Source | Method |
|------|--------|--------|
| Team efficiency/ratings | Barttorvik | Scrape with requests + pandas |
| Box scores & results | ESPN API | `cbbpy` or direct calls |
| Play-by-play | ESPN API | `cbbpy` |
| Historical odds | SBRO | Download Excel files |
| Current odds | The Odds API (free tier) | REST API |

#### Budget: ~$50/year
| Need | Source | Method |
|------|--------|--------|
| Team efficiency/ratings | **KenPom ($20/yr)** | `kenpompy` |
| Box scores & results | ESPN API | `cbbpy` |
| Play-by-play | ESPN API | `cbbpy` |
| Historical odds | SBRO | Download Excel files |
| Current odds | **The Odds API (~$20/mo)** | REST API |
| Supplementary ratings | Barttorvik (free) | Scrape |

### Data Update Frequency

| Source | Update Speed |
|--------|-------------|
| ESPN API (live) | ~10-15 seconds |
| ESPN API (box scores) | ~30-60 min post-game |
| NBA.com / nba_api | ~15-30 min post-game |
| Sports Reference | 1-2 hours post-game |
| KenPom | Next morning (~6-8 AM ET) |
| Barttorvik | Within a few hours |
| The Odds API | Real-time (poll frequency per tier) |

---

## 5. Strategy & Bankroll

### Which Markets to Target (Softest to Sharpest)

1. **Player props** — softest, lowest limits (~$250-500 max). Books set algorithmically, especially for college.
2. **First half totals** — under-modeled by books
3. **Full game totals** — pace variation across 363 teams creates mispricing opportunities
4. **First half spreads** — derived algorithmically from full-game lines
5. **Full game spreads (low-profile games)** — mid-major, early season
6. **Full game spreads (high-profile games)** — sharpest market

**Totals are widely considered the best market for quantitative models in college basketball.** Pace varies enormously (60-75+ possessions/game), and when two teams with unusual pace profiles meet, books sometimes misprice.

### Bankroll Management: Kelly Criterion

**Full Kelly:**
```
f* = (b*p - q) / b
```
Where: f* = fraction to wager, b = net odds, p = win probability, q = 1-p

**Example**: 55% edge at -110 odds (b = 0.909):
```
f* = (0.909 * 0.55 - 0.45) / 0.909 = 5.5% of bankroll
```

**In practice, use fractional Kelly:**
- **Full Kelly**: Maximizes growth but 30-50% drawdowns are expected
- **Half Kelly**: Sacrifices ~25% growth rate, dramatically cuts variance
- **Quarter Kelly**: Conservative, recommended for starting out or uncertain edge estimates

**Unit sizing alternative:**
- 1 unit = 1-2% of bankroll
- Standard bets: 1 unit
- Strong conviction: 2-3 units
- Maximum: 5 units (rare)

### Expected Drawdowns (2-3% ROI Edge)

- 10-15% drawdown: Will happen multiple times per season
- 20-25% drawdown: Likely at least once per season
- 30%+ drawdown: Possible even with a real edge

**Never change your model during a drawdown unless you have evidence-based reasons.** Over-reacting to variance is the #1 killer of profitable bettors.

### Bets Per Season

| Threshold | Bets/Season | Per-Bet ROI | Notes |
|-----------|-------------|-------------|-------|
| Aggressive (1+ pt edge) | 500-1,000 | Lower | More volume, more variance smoothing |
| Moderate (2+ pt edge) | 150-400 | Moderate | Good balance |
| Conservative (3+ pt edge) | 50-150 | Higher | Requires larger bankroll for variance |

---

## 6. Backtesting

### Walk-Forward Validation (Gold Standard)

1. Train on seasons 2015-2020
2. Validate on 2020-21 (predicting each game using only pre-game data)
3. Add 2020-21 to training, validate on 2021-22
4. Repeat through all available seasons

**Critical rules:**
- Never use future information for any prediction
- Account for line availability (use historical odds from SBRO)
- Simulate realistic execution (0.5-1 point slippage or use closing lines)

### Closing Line Value (CLV) — The Best Edge Metric

- **Definition**: Whether the line moves in your direction between your bet and game time
- **Example**: Bet Team A -3, line closes at -4.5 → you got 1.5 points of CLV
- **Why it matters**: CLV is less noisy than raw P&L. A bettor who consistently gets positive CLV is almost certainly a long-term winner.
- **Target**: Even 0.5-1 point average CLV on spreads is a meaningful edge

### Statistical Significance

At 95% confidence, to distinguish your win rate from 50%:
- 53% observed → ~2,500 bets needed
- 55% observed → ~600 bets needed
- 57% observed → ~250 bets needed

**Practical implication**: You may need 2-3 full seasons before you can confirm your edge. Use CLV-based testing for faster signal (continuous variable gives more power per observation).

---

## 7. Realistic Expectations

### Achievable ROI

| Level | ROI | Notes |
|-------|-----|-------|
| Typical public bettor | -5% to -10% | Roughly the vig |
| Decent model, poor execution | -2% to +1% | Close to breakeven |
| Good model, good execution | +1% to +4% | Consistently profitable |
| Excellent model, optimal execution | +3% to +7% | Professional-level |
| Soft markets (props, exotics) | +5% to +12% | Low volume, low limits |

**At -110 odds, breakeven is 52.38% ATS.** A realistic edge for a good model: 53-55% ATS. At 55% ATS, expected profit ≈ 4.5 cents per dollar wagered.

### Capital Requirements

| Bankroll | Avg Bet (Quarter Kelly, 2% edge) | Annual Profit (300 bets) |
|----------|----------------------------------|-------------------------|
| $5,000 | $25-50 | $150-300 |
| $25,000 | $125-250 | $750-1,500 |
| $100,000+ | $500-1,000 | $3,000-6,000+ |

### Time Investment

- **Initial model development**: 100-300+ hours
- **Daily maintenance during season**: 1-3 hours/day
- **Offseason**: 50-100+ hours for refinement, transfer portal, preseason projections
- **Total annual**: ~400-700+ hours

### The Honest Bottom Line

College basketball is likely the **best major US sport for a new quantitative bettor** for structural reasons. But "best opportunity" ≠ "easy money." The market has gotten significantly sharper over the past decade.

**The right mindset**: Treat this as a challenging intellectual pursuit with potential modest financial upside. If you enjoy data science, statistical modeling, and basketball, the process itself is rewarding.

**The real edge often comes from game selection, not prediction**: Rather than betting every game, identify games where your model's probability diverges most from market-implied probability.

---

## 8. Legal

### US State-by-State
- 30+ states have legal online sports betting (as of early 2025)
- Legal and online: AZ, CO, CT, IL, IN, IA, KS, KY, LA, ME, MD, MA, MI, NH, NJ, NY, NC, OH, OR, PA, RI, TN, VT, VA, DC, WV, WY
- Not yet legal: CA, TX, GA, FL (contested)
- **Laws change frequently** — verify current status

### Using Models/Algorithms
- **Legal everywhere that sports betting is legal.** No jurisdiction prohibits using statistical models.
- Distinct from insider information (using non-public material info could violate gaming regulations)
- Books CAN limit/ban you for winning — this is legal on their part and is the #1 practical obstacle for sharp bettors
- Exchanges don't have this problem

### Tax Implications
- **All winnings are taxable income** (regardless of whether the book reports them)
- Gambling losses deductible only up to winnings amount (must itemize)
- Professional gamblers (IRC Section 162) can deduct related expenses — high bar to qualify
- **Keep detailed records**: date, type, amount wagered, won/lost, book used

---

## 9. References

### Books
- **"Trading Bases" by Joe Peta** — Building a sports betting model from scratch (baseball, but methodology translates)
- **"The Signal and the Noise" by Nate Silver** — Sports betting chapter on difficulty of sustaining edge
- **"Calculated Bets" by Steven Skiena** — Academic approach to prediction model building
- **"Statistical Sports Models in Excel" by Andrew Mack** — Practical guide before scaling to code

### Academic Papers
- Sauer (1998), "The Economics of Wagering Markets" — *Journal of Economic Literature*. Betting markets are remarkably efficient.
- Levitt (2004), *Journal of Political Economy* — Books don't just balance action; they set lines based on sophisticated models.
- Paul and Weinbach (2005, 2007) — Point spread market efficiency in college basketball. Found inefficiencies in public biases.
- Zimmermann, Moorthy, and Shi (2013) — "Predicting College Basketball Match Outcomes Using ML". Logistic regression competitive with complex methods.
- Lopez and Matthews (2015) — Bayesian state-space model for NCAA team strengths.
- Manner (2016) — "Modeling and Forecasting NBA Basketball Games". Probit model with efficiency metrics outperformed complex approaches.
- Nichols (2014) — Travel distance creates exploitable biases in lines.

### Key People to Follow
- **Ken Pomeroy** (KenPom) — College basketball efficiency metrics pioneer
- **Bart Torvik** — Free alternative analytics with additional data points
- **Haralabos Voulgaris** — Famous basketball bettor (profiled in "The Odds")
- **Rufus Peabody** — Professional bettor, public about methodology
- **Ed Miller & Matthew Davidow** — Sports betting modeling framework

### Websites
- kenpom.com — College basketball efficiency ratings ($20/yr)
- barttorvik.com — Free college basketball analytics
- evanmiya.com — Player-level advanced metrics
- sportsbookreviewsonline.com — Free historical odds data
- the-odds-api.com — Real-time odds API
