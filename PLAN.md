# Basketball ML Betting: Phased Deployment Plan (Canada)

> Created February 2026. Targeting the 2026-27 NCAAB season as first live season.

---

## Overview

Six phases spanning ~14 months, from data infrastructure setup (now) through a full live betting season (2026-27). Each phase has clear deliverables, go/no-go criteria, and estimated time investment. The plan is designed to be killed early if evidence says the edge isn't there.

**Total estimated time investment:** 400-600 hours over 14 months
**Total estimated cash investment:** $500-1,500 CAD (data subscriptions, tax consultation) before any bankroll
**Target first live bet:** January 2027 (after 2 months of paper trading)

---

## Phase 0: Foundation & Infrastructure
**Timeline:** February - April 2026 (now through end of current season)
**Hours:** 60-80
**Cost:** ~$50 (KenPom subscription)

### Goals
- Working data pipeline that can collect, store, and retrieve NCAAB game data
- Historical dataset assembled for backtesting (2015-2026)
- Platform accounts opened and STX liquidity evaluated
- Paper trade the final weeks of the 2025-26 season + March Madness with a naive model

### Deliverables

#### 0.1 Environment Setup
- [ ] Python environment with core packages: `cbbpy`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `requests`
- [ ] SQLite feature store with temporal query patterns (no lookahead bias by design)
- [ ] Project directory structure matching CLAUDE.md spec
- [ ] Git repository initialized

#### 0.2 Data Collection Pipeline
- [ ] `cbbpy` pulling box scores and play-by-play for current season games
- [ ] Barttorvik scraper (or cbbdata API) pulling team ratings, Four Factors, game logs
- [ ] The Odds API (free tier) pulling live NCAAB odds for today's games
- [ ] SBRO historical odds downloaded (2015-2026 closing lines)
- [ ] Daily scheduler that runs COLLECT at 6 AM ET automatically
- [ ] All data flowing into SQLite with proper date indexing

#### 0.3 Historical Dataset Assembly
- [ ] Game-level dataset: every D1 game from 2015-16 through 2025-26
- [ ] Per-game features: box score stats, Four Factors, pace, location, rest days
- [ ] Matched to historical closing lines (SBRO data)
- [ ] Data quality checks: missing values, duplicate games, impossible stat lines
- [ ] Feature computation verified against known sources (spot-check against KenPom/Barttorvik)

#### 0.4 Platform Setup
- [ ] STX account opened (Ontario)
- [ ] Pinnacle account opened
- [ ] STX liquidity evaluation: log available NCAAB markets daily for 2+ weeks
- [ ] Document: which games have STX markets, typical spread width, depth

#### 0.5 Naive Paper Trading (March Madness 2026)
- [ ] Build simplest possible model: logistic regression with home/away + KenPom AdjEM differential
- [ ] Generate predictions for remaining regular season + conference tournament + NCAA tournament games
- [ ] Log predictions vs closing lines vs outcomes (spreadsheet is fine)
- [ ] Do NOT bet real money. This is pipeline validation, not edge validation.

### Go/No-Go for Phase 1
- Data pipeline runs reliably for 14+ consecutive days without manual intervention
- Historical dataset has >90% game coverage for 2015-2026 seasons
- At least one betting platform account is active and funded (even if just the minimum deposit)

---

## Phase 1: Baseline Model & Backtesting
**Timeline:** May - July 2026 (offseason)
**Hours:** 80-120
**Cost:** ~$20 (KenPom renewal)

### Goals
- Rigorous logistic regression baseline with walk-forward validation
- Understand whether the Four Factors approach can beat the closing line historically
- Build the evaluation framework (CLV, calibration, log loss) that all future models will be measured against

### Deliverables

#### 1.1 Feature Engineering (One at a Time)
Build features incrementally, measuring marginal improvement at each step. For each feature added, record the change in log loss, calibration (ECE), and simulated CLV on the validation set.

**Feature sequence (per CLAUDE.md):**

| Step | Feature(s) | Expected Impact |
|------|-----------|----------------|
| 1 | Pace differential (raw) | Baseline tempo context |
| 2 | Net Rating differential (ORtg - DRtg) | Major jump — strongest single predictor |
| 3 | eFG% differential (off + def) | Moderate improvement |
| 4 | TOV% differential (off + def) | Small improvement |
| 5 | ORB% differential (off + def) | Small improvement |
| 6 | FTr differential (off + def) | Marginal improvement |
| 7 | Home court indicator | Moderate improvement |
| 8 | Rest days differential | Small improvement |
| 9 | Opponent-adjusted versions (ridge regression) | Moderate improvement — key step |

For each feature:
- [ ] Compute using only pre-game data (no lookahead)
- [ ] Use exponential decay weighting (alpha 0.93 for offense, 0.95 for defense, 0.90 for pace)
- [ ] Record marginal improvement in log loss on held-out season
- [ ] Stop adding features when improvement < 0.5% relative log loss reduction

#### 1.2 Walk-Forward Validation
- [ ] Train on 2015-2020, validate on 2020-21
- [ ] Train on 2015-2021, validate on 2021-22
- [ ] Train on 2015-2022, validate on 2022-23
- [ ] Train on 2015-2023, validate on 2023-24
- [ ] Train on 2015-2024, validate on 2024-25
- [ ] Train on 2015-2025, validate on 2025-26
- [ ] Report average and per-season: log loss, ECE, AUC, simulated CLV, simulated ROI at quarter-Kelly

#### 1.3 Market-Specific Models
- [ ] Evaluate: does the same model work for spreads AND totals, or do they need separate models?
- [ ] Totals model: pace features should be more important. Test totals-specific feature weighting.
- [ ] Spreads model: net rating and Four Factors should dominate.
- [ ] Decision: unified model or separate models for each market type

#### 1.4 Evaluation Framework
Build reusable evaluation code that will be used throughout all subsequent phases:
- [ ] `evaluate_calibration()` — ECE, reliability diagram, Brier score
- [ ] `evaluate_discrimination()` — log loss, AUC
- [ ] `simulate_betting()` — given predictions + historical odds, simulate P&L at various Kelly fractions
- [ ] `calculate_clv()` — compare model prediction time odds to closing line odds
- [ ] `significance_test()` — how many bets needed to confirm observed win rate at 95% confidence

#### 1.5 Baseline Results Document
- [ ] Write up: "How does a logistic regression with X features perform against the NCAAB closing line, 2020-2026?"
- [ ] Include: feature importance, per-season breakdown, which game types the model is best/worst at
- [ ] Honest assessment: is there evidence of edge, or is this model just recapitulating what the market already knows?

### Go/No-Go for Phase 2
- **Minimum:** Simulated CLV is positive (>0%) on at least 4 of 6 validation seasons
- **Target:** Average simulated CLV > 0.5% across all validation seasons
- **Kill criterion:** If average simulated CLV is negative, STOP. Go back to research. Do not proceed to Phase 2 with a model that can't beat the closing line in backtesting.

---

## Phase 2: Model Enhancement & Preseason Prep
**Timeline:** August - October 2026
**Hours:** 100-150
**Cost:** ~$250 ($20 KenPom + $20/mo Odds API + ~$150 EvanMiya + tax consultation)

### Goals
- Upgrade from logistic regression to GBM ensemble
- Add opponent adjustments and contextual features
- Build preseason projection system for the 2026-27 season
- Get tax consultation done before any live betting

### Deliverables

#### 2.1 Gradient Boosted Model
- [ ] LightGBM with same feature set as logistic regression baseline
- [ ] Hyperparameter tuning via walk-forward CV (NOT random k-fold):
  - `max_depth`: test 3, 4, 5
  - `learning_rate`: test 0.01, 0.03, 0.05
  - `n_estimators`: early stopping with 50-round patience
  - `reg_alpha` and `reg_lambda`: aggressive regularization
- [ ] Compare GBM vs logistic regression on same validation sets
- [ ] If GBM doesn't beat logistic regression by >1% log loss, keep logistic regression as primary

#### 2.2 Opponent Adjustment (Ridge Regression Method)
- [ ] Implement ridge regression opponent adjustment (per ML best practices research)
- [ ] Typical optimal alpha: 100-300, tune via cross-validation
- [ ] Compare against raw (unadjusted) features — measure marginal improvement
- [ ] Ensure adjustment uses only pre-game data (no temporal leakage)

#### 2.3 Contextual Features
Add one at a time, measure marginal improvement:
- [ ] Travel distance (geocode all D1 arenas, compute great-circle distance for away games)
- [ ] Schedule density (games in last 7 days, games in last 14 days)
- [ ] Conference vs non-conference indicator
- [ ] Early season indicator (first 10 games of season — when model confidence is lower)
- [ ] Rivalry/rivalry-adjacent flags (if data available)

#### 2.4 Calibration Layer
- [ ] Apply Platt scaling to GBM outputs (fit on validation fold, apply to test fold)
- [ ] Compare: isotonic regression vs Platt scaling vs no calibration
- [ ] Target: ECE < 0.015 on held-out data
- [ ] Build automated recalibration pipeline (weekly check, refit if ECE drifts)

#### 2.5 Ensemble
- [ ] Simple average of logistic regression + calibrated GBM probabilities
- [ ] Test: does ensemble beat both individual models? (it should, based on 2025 research)
- [ ] If yes, this becomes the production model
- [ ] If no, use whichever individual model has better calibration

#### 2.6 Preseason Projection System (for 2026-27)
- [ ] Data sources: Barttorvik preseason rankings, KenPom preseason ratings, EvanMiya BPR
- [ ] Returning production: what % of minutes/points/BPR return from last season
- [ ] Transfer portal: incoming transfer BPR ratings (EvanMiya tracks this)
- [ ] Recruit talent: 247Sports composite rankings → approximate BPR contribution
- [ ] Blending formula: `preseason_rating = f(returning_production, transfer_impact, recruit_talent, prior_year * regression_coefficient)`
- [ ] Validate: how well do preseason projections predict first-month outcomes historically?

#### 2.7 Tax & Legal Consultation
- [ ] Identify and engage a Canadian CPA/tax lawyer with gambling income experience
- [ ] Specific questions to ask:
  - "I use ML models to inform part-time sports betting. Am I at risk of business income classification?"
  - "What records should I keep?"
  - "At what profit level should I consider proactive reporting?"
- [ ] Document their advice
- [ ] Set up bet logging system that satisfies both tax and model evaluation needs

### Go/No-Go for Phase 3
- **Minimum:** Ensemble model simulated CLV > 0.5% average across walk-forward validation seasons
- **Target:** Simulated CLV > 1.0%, with positive CLV in 5+ of 6 seasons
- **Kill criterion:** If the GBM/ensemble doesn't meaningfully beat the logistic regression baseline (~2%+ improvement in log loss), the added complexity isn't justified. Proceed with the simpler model.
- **Tax consultation complete.** Do not proceed to live betting without professional tax advice.

---

## Phase 3: Paper Trading (Live Validation)
**Timeline:** November - December 2026 (first 2 months of 2026-27 season)
**Hours:** 40-60 (1-2 hours/day during season)
**Cost:** ~$40/mo (Odds API)

### Goals
- Validate the model in real-time against live lines
- Confirm the data pipeline works reliably in production
- Measure live CLV (the single most important metric)
- Identify execution issues before real money is at stake

### Deliverables

#### 3.1 Daily Pipeline in Production
- [ ] Pipeline runs automatically every morning at 6 AM ET
- [ ] Yesterday's results ingested, features updated, predictions generated by 7 AM
- [ ] Today's predictions compared against current Pinnacle/STX lines
- [ ] Bets identified where model edge > 3% (net of estimated commission)
- [ ] All predictions logged with timestamp, model probability, market implied probability, and line

#### 3.2 Paper Bet Tracking
- [ ] Log every "bet" the model would make (but don't place real bets)
- [ ] Record: game, market (spread/total), model prob, line at prediction time, closing line, result
- [ ] Track: simulated P&L at quarter-Kelly, CLV per bet, cumulative CLV
- [ ] Dashboard or spreadsheet updated daily

#### 3.3 Cold Start Monitoring
- [ ] First 2-3 weeks: model relies heavily on preseason priors (80% prior, 20% observed)
- [ ] Track: does the model's confidence appropriately increase as games are played?
- [ ] Track: are preseason projections reasonable? (compare model's team rankings to KenPom/Barttorvik)
- [ ] Flag any teams where preseason projection is clearly wrong (transfer portal misses, coaching changes)

#### 3.4 STX Liquidity Logging (Continued)
- [ ] For every game the model flags as a bet, check if STX has an active market
- [ ] Log: STX availability (yes/no), best available price, depth, time of check
- [ ] Build a realistic picture of what % of model bets could actually be placed on STX

#### 3.5 Two-Month Review
At the end of December 2026, comprehensive review:
- [ ] Total paper bets placed: target 80-150
- [ ] Average CLV: positive or negative?
- [ ] Calibration: are predicted probabilities matching observed frequencies?
- [ ] Pipeline reliability: how many days did the pipeline fail? What caused failures?
- [ ] STX coverage: what % of flagged bets had STX liquidity?
- [ ] Any bugs, data quality issues, or unexpected model behaviors?

### Go/No-Go for Phase 4
- **Minimum:** CLV > 0% over 100+ paper bets AND pipeline ran successfully >90% of days
- **Target:** CLV > 0.5% over 100+ paper bets, ECE < 0.02
- **Kill criterion:** If CLV is negative over 100+ paper bets, DO NOT go live. Return to Phase 1/2. The model is not adding value.
- **Delay criterion:** If pipeline reliability < 90%, fix infrastructure before going live. Money depends on this pipeline working.

---

## Phase 4: Live Betting (Conservative)
**Timeline:** January - April 2027 (second half of 2026-27 season + March Madness)
**Hours:** 50-70 (1-2 hours/day during season)
**Cost:** Bankroll dependent (see below)

### Goals
- Place real bets with real money using the validated model
- Confirm that live CLV matches paper trading CLV
- Build a track record of 200+ live bets for statistical evaluation
- Survive variance without blowing up the bankroll

### Betting Parameters

#### Bankroll & Sizing
| Bankroll | Unit Size (1% of bankroll) | Quarter-Kelly Max Bet | Season Target Bets |
|----------|---------------------------|----------------------|-------------------|
| $5,000 CAD | $50 | $25-75 | 200-400 |
| $10,000 CAD | $100 | $50-150 | 200-400 |
| $25,000 CAD | $250 | $125-375 | 200-400 |

- **Start with quarter-Kelly.** This is non-negotiable for the first live season.
- **Minimum edge to bet:** 3% net (after estimated 2% exchange commission on STX, or 2.5% implied vig on Pinnacle)
- **Maximum single bet:** 3% of bankroll regardless of model confidence
- **No parlays.** Single bets only.

#### Execution Priority
1. **STX** — if market exists with reasonable depth, bet here first (P2P, better odds potential)
2. **Pinnacle** — primary venue for games STX doesn't cover
3. **SX Bet** — supplementary for games where neither has good pricing

#### Markets to Target (in priority order)
1. **Full game totals** — pace mismatch games (model's strongest signal, softest market)
2. **Full game spreads (small conference)** — under-covered, lines set by algorithm
3. **Full game spreads (early season)** — softer lines before market has learned
4. **First half totals** — if model shows edge and platform offers the market

### Deliverables

#### 4.1 Live Bet Execution
- [ ] Place bets following model recommendations, within sizing rules above
- [ ] Log every bet: timestamp, platform, game, market, line, stake, model probability, result
- [ ] Track running bankroll daily
- [ ] Never deviate from the model. No "feel" bets.

#### 4.2 CLV Tracking (Live)
- [ ] For every bet placed, record the closing line
- [ ] Calculate CLV: did the line move in your direction after you bet?
- [ ] Weekly CLV summary: average CLV, # of positive CLV bets, # of negative CLV bets
- [ ] This is the most important metric. Win rate is noisy. CLV is signal.

#### 4.3 Variance Management
- [ ] Expected drawdowns at quarter-Kelly with 2-3% edge:
  - 10-15% drawdown: will happen, stay the course
  - 20%+ drawdown: review model, but do NOT change unless evidence-based reason exists
  - 30%+ drawdown: pause betting for 1 week, re-run backtests, check for data/pipeline issues
- [ ] **Hard stop:** If bankroll drops 40% from peak, stop all betting and conduct full review

#### 4.4 Monthly Review
At the end of each month:
- [ ] Bets placed, win rate, CLV, P&L, ROI
- [ ] Calibration check (ECE on last 100 predictions)
- [ ] Model drift check
- [ ] Any execution issues (STX liquidity, Pinnacle limiting, etc.)

#### 4.5 End-of-Season Review (April 2027)
- [ ] Total live bets: target 200-400
- [ ] Overall CLV: positive or negative?
- [ ] Overall ROI: positive or negative?
- [ ] Comparison: live results vs paper trading results vs backtest results
- [ ] Honest assessment: is there a real edge, or is this explainable by variance?
- [ ] Statistical significance test: given observed win rate over N bets, can we reject the null hypothesis of no edge?

### Go/No-Go for Phase 5
- **Minimum:** CLV > 0% over 200+ live bets (even if P&L is negative due to variance)
- **Target:** CLV > 0.5%, positive P&L, pipeline reliable >95% of days
- **Kill criterion:** If CLV is negative over 200+ live bets, the model does not have an edge in live markets. Accept this honestly.
- **Scale criterion:** If CLV > 1.0% over 200+ bets, consider increasing to half-Kelly in Phase 5

---

## Phase 5: Scale & Optimize (2027-28 Season)
**Timeline:** May 2027 - April 2028
**Hours:** 200-300 (offseason development + daily season operation)
**Cost:** ~$500/year (subscriptions) + bankroll

### Goals
- Scale bet sizing if edge is confirmed (move from quarter-Kelly to half-Kelly)
- Add model sophistication: Bayesian hierarchical model, player-level features
- Expand market coverage: first half totals, conference tournaments
- Build toward statistical significance (cumulative 600+ bets)

### Deliverables

#### 5.1 Model Upgrades (Offseason: May-October 2027)
- [ ] Bayesian hierarchical model (PyMC5): conference → team structure for 363 teams
  - Posterior uncertainty estimates for Kelly sizing (replace point estimates with distributions)
  - Partial pooling across conferences (small conference teams borrow strength from conference average)
- [ ] Player availability feature: use EvanMiya BPR to quantify injury/absence impact
  - `impact = absent_player_BPR * (minutes_share / 200) * replacement_quality_factor`
- [ ] Transfer portal model: preseason ratings based on incoming/outgoing player-level data
- [ ] Team embeddings (optional, if Bayesian model doesn't capture enough structure)

#### 5.2 Bet Sizing Upgrade
If Phase 4 confirmed CLV > 0.5%:
- [ ] Move from quarter-Kelly to half-Kelly
- [ ] Use Bayesian posterior uncertainty to adjust Kelly fraction per bet:
  - High-confidence predictions: closer to half-Kelly
  - Uncertain predictions: closer to quarter-Kelly
- [ ] Implement dynamic bankroll tracking (Kelly recalculated based on current bankroll, not initial)

#### 5.3 Market Expansion
- [ ] First half totals: build separate model or adjust full-game model
- [ ] Conference tournament betting: small conference auto-bid tournaments (high inefficiency)
- [ ] March Madness: selective betting on under-covered first-round matchups
- [ ] Monitor Alberta market launch: if STX expands, evaluate new liquidity

#### 5.4 Cumulative Edge Assessment
- [ ] 600+ total bets across Phase 4 + Phase 5 (target for statistical significance)
- [ ] Formal hypothesis test: "Is my observed win rate significantly different from breakeven?"
- [ ] CLV trend analysis: is the edge stable, growing, or shrinking?
- [ ] Comparison against naive strategies (always bet the dog, always bet the under, etc.)

---

## Key Decision Points Summary

| Checkpoint | Timing | Go Criterion | Kill Criterion |
|-----------|--------|-------------|----------------|
| Phase 0 → 1 | April 2026 | Pipeline works, data assembled | Pipeline unreliable after 3+ attempts to fix |
| Phase 1 → 2 | July 2026 | Simulated CLV > 0% on 4+ of 6 seasons | Simulated CLV negative average |
| Phase 2 → 3 | October 2026 | Ensemble CLV > 0.5%, tax consultation done | No improvement over baseline |
| Phase 3 → 4 | December 2026 | Paper CLV > 0% on 100+ bets | Paper CLV negative on 100+ bets |
| Phase 4 → 5 | April 2027 | Live CLV > 0% on 200+ bets | Live CLV negative on 200+ bets |
| Continue | April 2028 | Cumulative CLV > 0.5% on 600+ bets | Edge disappearing or negative |

**The kill criteria are as important as the go criteria.** The biggest risk in this project is not losing money — it's spending hundreds of hours building something that doesn't work and being too attached to admit it.

---

## Budget Summary

### Development Costs (Pre-Bankroll)

| Item | Cost | Timing |
|------|------|--------|
| KenPom subscription | $20 CAD/year | Phase 0 |
| EvanMiya subscription | $15 CAD/season | Phase 2 |
| The Odds API | $20-40 CAD/month (during season, ~6 months) | Phase 2+ |
| Tax consultation (CPA) | $300-500 CAD (one-time) | Phase 2 |
| **Total pre-bankroll** | **~$500-700 CAD** | |

### Bankroll Requirements

| Risk Tolerance | Starting Bankroll | Expected Annual Profit (2-3% edge, quarter-Kelly, 300 bets) |
|---------------|-------------------|-------------------------------------------------------------|
| Conservative | $5,000 CAD | $100-250 CAD |
| Moderate | $10,000 CAD | $200-500 CAD |
| Serious | $25,000 CAD | $500-1,250 CAD |

**Honest framing:** At quarter-Kelly with a 2-3% edge, you need $25K+ bankroll for the profits to exceed the cost of your time. Treat the first season (Phase 4) as a learning investment, not an income source.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| No real edge exists | Medium | High | Kill criteria at every phase. CLV-based evaluation. |
| STX liquidity too thin | High | Medium | Pinnacle as primary. STX supplementary. Don't over-index on P2P. |
| CRA reclassifies as business | Low | Medium | Tax consultation. Keep day job. Modest bet sizes. |
| Data pipeline fails during season | Medium | Medium | Automated monitoring. Manual fallback process documented. |
| Pinnacle limits account | Low | Low | Pinnacle's "Winners Welcome" policy. Extremely rare for NCAAB volume. |
| Model overfits backtest | Medium | High | Walk-forward only. Paper trading. CLV as primary metric. |
| Bankroll variance causes ruin | Low | High | Quarter-Kelly. Hard stop at -40%. Never chase losses. |
| Market gets sharper over time | Medium | Medium | Focus on structural inefficiencies (small conference, early season). Adapt. |

---

## Appendix: Season Calendar Alignment

| Month | NCAAB Activity | Project Activity |
|-------|---------------|-----------------|
| **Feb 2026** | Regular season | **Phase 0**: Build pipeline, collect live data |
| **Mar 2026** | Conference tournaments + March Madness | **Phase 0**: Paper trade March Madness with naive model |
| **Apr 2026** | Season ends | **Phase 0 → 1**: Assemble full historical dataset |
| **May-Jul 2026** | Offseason (transfer portal active) | **Phase 1**: Baseline model, backtesting, evaluation framework |
| **Aug-Oct 2026** | Offseason (preseason projections) | **Phase 2**: GBM, opponent adjustment, preseason system, tax consult |
| **Nov 2026** | Season starts | **Phase 3**: Paper trading begins (early season = softer lines) |
| **Dec 2026** | Non-conference ends, conference play begins | **Phase 3 → 4**: Review paper trading, go/no-go for live |
| **Jan 2027** | Full conference play | **Phase 4**: Live betting begins (conservative) |
| **Feb 2027** | Conference play | **Phase 4**: Live betting continues |
| **Mar 2027** | Conference tournaments + March Madness | **Phase 4**: March Madness betting (selective) |
| **Apr 2027** | Season ends | **Phase 4 → 5**: End-of-season review |
| **May-Oct 2027** | Offseason | **Phase 5**: Model upgrades, Bayesian model, player features |
| **Nov 2027 - Apr 2028** | 2027-28 season | **Phase 5**: Scaled live betting (if edge confirmed) |
