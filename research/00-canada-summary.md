# Basketball ML Betting: Canadian Market Research Summary

> Compiled February 2026. Cross-references detailed research in this directory.

---

## Executive Summary

This research adapts the basketball ML betting project (originally US-focused) for the **Canadian regulatory market**. The findings are broadly positive: Canada offers structural advantages over the US for this type of project, particularly around taxation. However, the peer-to-peer exchange landscape is significantly more limited.

### Key Takeaways

1. **Regulation is favorable.** Single-event betting is legal across Canada (Bill C-218, August 2021). Ontario has the most developed private market with 48+ licensed operators. Alberta's private market launches mid-2026. No restrictions on algorithmic/model-based betting exist anywhere in Canada.

2. **Tax advantage is significant.** Gambling winnings are tax-free for recreational bettors in Canada (unlike the US where all winnings are taxable). This is a major structural edge. However, running a systematic ML betting operation could risk CRA reclassification as "business income." Key case law (*Leblanc*, *Duhamel*) sets a high bar for this, but the 2025 *Fournier-Giguere* FCA decision shows it can happen in extreme cases.

3. **P2P exchange options are limited.** STX (Ontario) is the only regulated exchange in Canada. Betfair is blocked. US platforms (Sporttrade, Novig, Prophet) are unavailable. Crypto options (SX Bet, Overtime Markets) are accessible but unregulated. Pinnacle (sharp-friendly, available nationwide) is the best practical alternative despite not being a true exchange.

4. **NCAA basketball is fully available** with no college-sport restrictions in any province. This is an advantage over several US states that restrict college betting.

5. **ML best practices are well-established.** Logistic regression remains a strong baseline. GBMs (XGBoost/LightGBM) are the workhorse. Bayesian hierarchical models offer uncertainty quantification for Kelly sizing. Calibration matters more than raw accuracy for betting. Recent research (2025-2026) shows stacked ensembles and transformer architectures are competitive but not clearly superior to well-tuned GBMs on tabular data.

---

## Detailed Research Files

| File | Contents | Lines |
|------|----------|-------|
| `05-canada-exchanges.md` | P2P platforms available in Canada, platform comparison matrix, operational setup | ~500 |
| `05-canada-tax-legal.md` | CRA tax treatment, case law, algorithmic betting legality, data scraping, banking | ~540 |
| `05-deep-dive-ml-best-practices.md` | Model architectures, feature engineering, validation, pipeline design, 2025-2026 research | ~1,480 |

See also: `../RESEARCH.md` for the original US-focused comprehensive research.

---

## Canadian Regulatory Landscape

### Bill C-218 (Single-Event Betting)
- Passed June 2021, in force August 27, 2021
- Amended Criminal Code to allow single-event betting (previously only parlays were legal)
- Delegated all regulatory authority to provinces

### Provincial Market Status (February 2026)

| Province | Private Operators | Exchange Available | Status |
|----------|------------------|-------------------|--------|
| **Ontario** | Yes (48+ operators) | Yes (STX) | Mature since April 2022 |
| **Alberta** | Yes (pending) | Not yet | Bill 48 passed; launching mid-2026 |
| British Columbia | No (BCLC monopoly) | No | No plans to open |
| Quebec | No (Loto-Quebec monopoly) | No | May open eventually |
| Other provinces | No | No | Provincial monopolies |

### Best Province for This Project
**Ontario** by a wide margin: most operators, only legal exchange (STX), Pinnacle licensed, full NCAA basketball coverage, mature regulatory framework.

---

## P2P / Exchange Options for Canadians

### Regulated
- **STX** (Ontario only) -- Only legal P2P exchange. AGCO-licensed. NCAA basketball markets for major games. ~2-3% commission. Developing API. Thin liquidity for small-conference games.

### Sharp-Friendly (Not P2P but Doesn't Limit Winners)
- **Pinnacle** -- Available nationwide. "Winners Welcome" policy. Lowest margins (~2-3%). Comprehensive NCAA coverage. API access closed to public (July 2025) but bespoke access available for high-volume bettors.

### Crypto/Unregulated (Accessible but Grey Area)
- **SX Bet** -- Blockchain order book. Not explicitly blocked in Canada. API available. Limited NCAAB.
- **Overtime Markets** -- DeFi AMM. No KYC. Some NCAAB. 2-3% spread.
- **Azuro** -- DeFi protocol. Limited basketball.

### Not Available in Canada
- Betfair (blocked since 2016)
- Sporttrade, Novig, Prophet Exchange (US only)
- Kalshi (US CFTC only)
- Polymarket (banned in Ontario by OSC, April 2025)

### Recommended Platform Strategy
- **Ontario**: STX (primary exchange) + Pinnacle (coverage STX lacks) + SX Bet (supplementary crypto)
- **Outside Ontario**: Pinnacle (primary) + crypto platforms (supplementary)

---

## Tax & Legal Summary

### The Canadian Advantage
- **Recreational gamblers: winnings are TAX-FREE** (unlike US where all winnings are taxable at federal + state rates)
- **No withholding** on gambling winnings in Canada
- **No reporting requirement** for recreational winnings

### The Risk: Business Income Classification
Using an ML model systematically increases the risk CRA classifies your activity as a business. Key factors CRA evaluates:
1. Degree of organization and business-like conduct
2. Special knowledge/skill reducing element of chance
3. Profit intent vs. pleasure/entertainment
4. Frequency and extent of activity

### Case Law Protection
- **Leblanc (2006 TCC)**: Even $10-13M/year wagered using computer systems was NOT taxable
- **Duhamel (2022 TCC)**: WSOP winner's skill was NOT enough for business classification
- **Fournier-Giguere (2025 FCA)**: $1.45M poker income, sole income, full-time = taxable (the high-water mark)

### Practical Strategy to Maintain Tax-Free Status
1. Maintain a day job with substantial employment income
2. Keep betting part-time, not your livelihood
3. Keep bet sizes modest relative to total income
4. Do not advertise as a professional bettor
5. Consult a Canadian tax lawyer proactively

### Algorithmic Betting Legality
- **Fully legal** -- no Canadian law prohibits using ML models for betting decisions
- **Automated execution** may violate platform ToS (civil matter, not criminal)
- **Data scraping** of public sports statistics is low risk; respect ToS and rate limits

---

## ML Strategy Summary

### Recommended Model Stack
1. **Baseline**: Logistic regression with Four Factors differentials
2. **Primary**: LightGBM/XGBoost ensemble (max_depth 3-5, aggressive regularization)
3. **Advanced**: Bayesian hierarchical model (PyMC5) for uncertainty-aware Kelly sizing
4. **Evaluation ensemble**: Combine all three with calibrated probability averaging

### Key Feature Engineering Insights
- Use **exponential decay weighting** (alpha 0.93-0.95) instead of hard rolling windows
- **Opponent adjustment via ridge regression** is more stable than iterative KenPom method
- **Calibration > accuracy** for betting: use Platt scaling or isotonic regression
- **Cold start**: blend preseason priors with in-season data using `w = games / (games + k)`, k ~ 12
- **CLV (Closing Line Value)** is the gold-standard edge metric, not win rate

### Recent Research Highlights (2025-2026)
- Stacked ensembles (6 base learners + MLP meta) achieve 83% NBA prediction accuracy
- LSTM with Brier loss produces better-calibrated probabilities than Transformers for NCAAB
- Vegas spread error has *increased* (9.12 to 10.49 pts) due to 3-point variance -- game is harder to predict, not just lines getting sharper
- Kelly criterion + Bayesian posterior uncertainty provides formal justification for fractional Kelly

### Data Pipeline Architecture
```
Daily: COLLECT (6AM) -> PROCESS (6:30AM) -> PREDICT (7AM) -> EXECUTE -> EVALUATE (next day)
```
- Feature store with temporal ordering (no lookahead bias)
- SQLite for solo project, PostgreSQL for scaling
- Drift detection: retrain if ECE > 0.015 over rolling 100-prediction window

---

## Research Questions for Phase Planning

The following questions should inform the next phase (creating a phased deployment plan):

### Platform & Execution
1. **STX liquidity testing**: What is actual NCAAB market depth on STX? How many games per day have active markets? What are typical spreads?
2. **STX API status**: Is there a private/beta API? What are the terms for programmatic access?
3. **Pinnacle API**: What constitutes "high-volume" for bespoke API access? What are the thresholds?
4. **Multi-platform execution**: How to efficiently split bets across STX + Pinnacle + crypto to maximize coverage?
5. **Alberta timeline**: When exactly does Alberta's market launch, and will STX expand there?

### Data & Infrastructure
6. **cbbdata API evaluation**: How reliable is the Barttorvik API for daily automated updates? Rate limits? Data quality?
7. **The Odds API for CLV tracking**: What tier is needed for sufficient NCAAB coverage? Can we track opening → closing line movement?
8. **Historical data assembly**: What's the fastest path to a clean dataset of NCAAB game-level features + historical odds for backtesting (2015-2026)?
9. **Player availability data**: What's the best source for college basketball injury/suspension data that could be integrated into the pipeline?

### Model Development
10. **Baseline model benchmarking**: How does a simple logistic regression with 8 Four Factors (offense + defense) perform against the closing line on historical NCAAB data?
11. **Feature importance by market**: Do different features matter for spreads vs totals vs moneylines? Should we build separate models?
12. **Bayesian hierarchical model**: How to structure conference → team hierarchy for 363 D1 teams across 32 conferences? What priors work best?
13. **Transfer portal modeling**: How to quantify the impact of incoming/outgoing transfers on team ratings? What data sources track this?
14. **Ensemble combination**: How to weight the logistic regression, GBM, and Bayesian model outputs? Meta-learning vs simple averaging?

### Betting Strategy
15. **Market selection**: Which NCAAB markets (spreads, totals, first half, props) are most accessible on STX and Pinnacle in Canada?
16. **Optimal Kelly fraction**: Given model uncertainty in early development, what fractional Kelly (1/4, 1/3, 1/2) minimizes ruin risk?
17. **Minimum bankroll**: What bankroll is needed for quarter-Kelly sizing at 300+ bets/season with a 2-3% edge?
18. **Paper trading period**: How many bets should we paper trade before going live? What metrics must be met?

### Tax & Legal
19. **CPA engagement**: Identify a Canadian CPA/tax lawyer with gambling income experience to consult before going live
20. **Record-keeping system**: Design a bet logging system that supports both tax compliance and model evaluation
21. **Professional vs recreational classification**: At what annual profit level should we proactively classify as business income vs defending recreational status?

### Season Timing
22. **2026-27 season prep**: The NCAAB season starts in November. What's the minimum viable system to have ready by then?
23. **Preseason projections**: How to generate prior ratings for the 2026-27 season using transfer portal data, returning production, and recruit rankings?
24. **Early season vs late season strategy**: Should we bet more aggressively in November-December (softer lines) or be more conservative (less model confidence)?

---

## Recommended Next Steps

1. **Set up data infrastructure** -- Get cbbpy, The Odds API, and Barttorvik scraping working. Assemble historical dataset.
2. **Build and backtest baseline model** -- Logistic regression with Four Factors on 2015-2025 data. Walk-forward validation.
3. **Platform accounts** -- Open STX and Pinnacle accounts (Ontario). Evaluate STX liquidity on live NCAAB games.
4. **Tax consultation** -- Engage a Canadian CPA before placing any real bets.
5. **Paper trading** -- Run the model live for the remainder of the 2025-26 season (February-March 2026) without real money.
6. **Iterate and plan** -- Use paper trading results to inform the phased deployment plan for the 2026-27 season.
