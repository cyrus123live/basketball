# Peer-to-Peer Betting Exchanges for Canadian Bettors (February 2026)

> Research compiled February 13, 2026. Platform availability, regulations, and fee structures change frequently. Verify details before depositing funds.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Regulated Betting Exchanges Accessible from Canada](#2-regulated-betting-exchanges)
3. [Crypto / Decentralized Platforms](#3-crypto--decentralized-platforms)
4. [Canadian-Specific Platforms and Regulation](#4-canadian-specific-platforms)
5. [Betfair Access via Broker Services](#5-betfair-access-via-broker-services)
6. [Sharp-Friendly Traditional Sportsbooks (Fallback)](#6-sharp-friendly-traditional-sportsbooks)
7. [Platform Comparison Matrix](#7-platform-comparison-matrix)
8. [Practical Recommendations for a Canadian ML Bettor](#8-practical-recommendations)
9. [Sources](#9-sources)

---

## 1. Executive Summary

**The situation for Canadian bettors seeking P2P/exchange betting is significantly worse than for US or UK bettors.** Here is the reality as of February 2026:

- **Betfair** is blocked in Canada (since 2016). Indirect access is possible through betting brokers (Orbit Exchange, Piwi247) but comes with added cost, complexity, and counterparty risk.
- **US exchanges** (Sporttrade, Novig, Prophet Exchange/ProphetX) are US-only. None accept Canadian residents.
- **Kalshi** is US-only (CFTC jurisdiction does not extend to Canada).
- **STX (Sports Trading Exchange)** is the only regulated betting exchange in Canada. It is licensed in Ontario and offers NCAA basketball markets. This is the single most important platform for this project.
- **Crypto platforms** (SX Bet, Overtime Markets, Azuro) are accessible from Canada with varying degrees of legal grey area. Polymarket is explicitly banned in Ontario but technically accessible from other provinces.
- **Pinnacle** (not an exchange but sharp-friendly) accepts Canadian bettors, does not limit winners, and offers college basketball with high limits.
- **Alberta** is launching a regulated iGaming market in spring/summer 2026, potentially opening the door for additional exchange operators.

**Bottom line for this project:** A Canadian bettor building an ML basketball betting system should plan around STX (Ontario) as the primary exchange, Pinnacle as the sharp-friendly sportsbook fallback, and crypto platforms (SX Bet, Overtime Markets) as supplementary options for additional liquidity.

---

## 2. Regulated Betting Exchanges

### 2.1 Betfair Exchange

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **No.** Blocked since 2016. |
| **Reason** | Canadian law prohibits betting exchange offers (odds trading). Betfair voluntarily ceased Canadian operations. |
| **Workaround** | Betfair white-label brokers (see Section 5) |
| **NCAA basketball** | Limited to major games and March Madness on Betfair proper |
| **Commission** | 5% on net winnings (down to 2% for high volume) |
| **API** | Mature (Betfair API-NG) -- industry gold standard |

Betfair is the world's largest and most liquid betting exchange with the most mature API for algorithmic betting. Its unavailability in Canada is the single biggest gap for this project. The only route to Betfair liquidity is through white-label brokers (see Section 5), which adds a layer of complexity and counterparty risk.

### 2.2 Smarkets

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **Effectively no.** |
| **Detail** | Despite having CAD listed on their registration page, Smarkets support has confirmed they do not accept Canadian players. Canada is not listed in the country dropdown during registration. |
| **Commission** | 2% on net winnings |
| **API** | Robust REST API |
| **NCAA basketball** | Limited markets |

### 2.3 Sporttrade

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **No.** US-only. |
| **US states** | Arizona, Colorado, Iowa, New Jersey, Virginia |
| **Current status** | Applied for CFTC registration as a designated contract market (DCM) in February 2026, which could broaden US reach but would not extend to Canada |
| **NCAA basketball** | Yes (spreads, moneylines, totals) |
| **Commission** | ~2% on net winnings |

### 2.4 Novig

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **No.** US-only. |
| **US reach** | Available in 40+ states via sweepstakes model |
| **Current status** | Registered with CFTC in January 2026 to supply sports event contracts |
| **NCAA basketball** | Yes |
| **Commission** | ~1% embedded vig |
| **API** | No (consumer-focused UX) |

### 2.5 Prophet Exchange (ProphetX)

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **No.** US-only. |
| **US reach** | Available in ~40 states (relaunched September 2024 with sweepstakes model) |
| **NCAA basketball** | Major conferences and tournaments |
| **Commission** | 2-3% on net winnings |

### 2.6 Kalshi

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **No.** US residents only. |
| **Reason** | CFTC jurisdiction is strictly US-based. Canada has no equivalent regulatory body for prediction markets. Binary options (which prediction market contracts resemble) have been banned in Canada since 2017 by the Canadian Securities Administrators (CSA). |
| **Sports markets** | Expanding into NBA, NFL, college sports. 85+ active NBA markets as of February 2026. |

---

## 3. Crypto / Decentralized Platforms

### Legal Context for Canada

The legal status of crypto betting in Canada is a grey area with limited enforcement:

- **Binary options** are banned nationwide under CSA regulation MI 91-102 (since 2017). Prediction market contracts may fall under this prohibition.
- **Ontario** has the strictest enforcement. The Ontario Securities Commission (OSC) settled with Polymarket in April 2025, imposing a $200,000 CAD penalty and permanently banning Ontario residents.
- **Other provinces** have not taken comparable enforcement action. Crypto betting platforms are technically accessible but exist in regulatory limbo.
- **No Canadian federal law** explicitly criminalizes individual bettors using offshore or decentralized platforms, but platforms themselves may violate Canadian securities or gaming law.

### 3.1 Polymarket

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **Ontario: Explicitly banned.** Other provinces: Technically accessible but legally grey. |
| **OSC settlement** | April 2025: $200,000 penalty + $25,000 investigation costs. Permanent ban for Ontario residents. |
| **Geo-blocking** | Ontario IP addresses blocked. Improved detection since December 2025. |
| **NCAA basketball** | **Yes.** 853 active NCAA markets, $2.1M+ trading volume. Game-by-game NCAAB markets, props, conference championships, tournament winner. |
| **NBA** | 85+ active markets. Game-by-game markets, MVP, championship futures. |
| **Commission** | ~0% (revenue from market maker spreads) |
| **KYC** | Required for US; minimal for non-US |
| **Deposit/Withdrawal** | USDC on Polygon. No CAD support. |
| **API** | On-chain (Polygon CLOB). Well-documented. |

**Assessment for this project:** Polymarket has the best college basketball coverage of any crypto platform. If you are outside Ontario, it is technically usable but carries regulatory risk. The OSC precedent means other provinces could follow. No CAD support -- must use USDC.

### 3.2 SX Bet

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **Likely yes** (Canada not listed as restricted in available documentation). US is explicitly blocked. |
| **Platform** | Largest blockchain sports betting exchange by volume. Peer-to-peer order book on SX Network (migrating to Arbitrum). |
| **NCAA basketball** | Some NCAAB markets (primarily major games) |
| **NBA** | Yes |
| **Commission** | 2-4% on net winnings |
| **KYC** | No mandatory KYC |
| **Deposit/Withdrawal** | Crypto (USDC, ETH). No CAD support. |
| **API** | Yes -- REST API + on-chain. Documented at api.docs.sx.bet |
| **Operator** | CRGC Management Ltd (Costa Rica), licensed under Anjouan Computer Gaming Licensing Act |

**Assessment:** SX Bet is the most exchange-like crypto platform with a proper order book (not AMM). The API enables algorithmic betting. However, liquidity for NCAAB is thin outside major games. Counterparty risk is higher than regulated platforms given the offshore licensing.

### 3.3 Overtime Markets

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **Yes.** No geographic restrictions or country blocking. No KYC required. |
| **Platform** | AMM (Automated Market Maker) built on Thales Protocol. Deployed on Optimism, Arbitrum, and Base. |
| **NCAA basketball** | Some NCAAB markets |
| **NBA** | Yes |
| **Fees** | 2-3% AMM spread (embedded in odds) |
| **KYC** | None |
| **Deposit/Withdrawal** | Crypto (USDC, sUSD on supported chains). No CAD. |
| **API** | On-chain smart contracts. Fully permissionless. |
| **Liquidity** | $3M+ in volume during beta; $200M+ protocol total. Pool-based liquidity (no counterparty matching needed). |

**Assessment:** Overtime's AMM model means you always have liquidity (no need to wait for counterparty matching), but the spread is wider than an order book exchange. Good as a supplementary platform. The "no restrictions, no KYC" positioning means it operates in a fully unregulated manner -- use at your own risk.

### 3.4 Azuro Protocol

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **Yes.** Permissionless infrastructure protocol -- no geo-blocking at the protocol level. Individual frontends built on Azuro may have their own restrictions. |
| **Platform** | Infrastructure/liquidity protocol for on-chain predictions. 27+ applications built on top. "Liquidity Tree" design. |
| **Basketball** | 18.6% of total volume is basketball (second largest after football at 69.4%) |
| **NCAA basketball** | Limited NCAAB coverage |
| **Commission** | 3-5% spread (varies by frontend application) |
| **KYC** | None at protocol level |
| **Deposit/Withdrawal** | Crypto. Varies by frontend. No CAD. |
| **API** | On-chain smart contracts (EVM-compatible). Subgraph API for data. |
| **Liquidity** | $358M total prediction volume. 4,525 liquidity providers. |

**Assessment:** Azuro is more infrastructure than a betting platform you interact with directly. You would use one of the 27+ frontends built on Azuro. Basketball is a meaningful share of volume but NCAAB is thin. The peer-to-pool model means liquidity is always available but odds may not be as sharp as a proper order book.

### 3.5 Other Crypto Platforms Worth Monitoring

- **Drift BET** (Solana): Very limited sports markets, extremely low fees (0.05-0.1%). Worth watching but not yet viable for basketball.
- **Pred X**: Cross-chain prediction market. Early stage.

---

## 4. Canadian-Specific Platforms

### 4.1 STX (Sports Trading Exchange) -- THE KEY PLATFORM

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **Yes -- Ontario only.** Licensed by AGCO (Alcohol and Gaming Commission of Ontario). |
| **Platform type** | Betting exchange (peer-to-peer order book). First and only licensed betting exchange in Canada. |
| **How it works** | Market participants set odds. Trades occur when buyer and seller agree on price. You can both back and lay outcomes. |
| **NCAA basketball** | **Yes.** Spreads and moneylines for major matchups. March Madness coverage. Props are rare. |
| **NBA** | Yes. Markets for every game during the season. Raptors games draw particularly strong action. |
| **Commission** | Up to 3% on winning bets. Exact rate depends on Rewards Program tier. |
| **Deposit methods** | Credit/debit card, bank transfer, Interac, MuchBetter, PayPal |
| **Currency** | **CAD** |
| **Withdrawal** | Bank account (3-5 business days). No minimum withdrawal. No fees. |
| **API** | **Yes.** Trading API documented at wiki.stxapp.io/en/trading-api. Market Info APIs available without authentication. Limit orders for market making; market orders for taking liquidity. API described as "strongly hardened against flood attacks." |
| **KYC** | Full KYC required (Ontario regulatory requirement) |
| **Regulator** | AGCO + audited by Gaming Labs International |
| **Ontario residency** | Required. Must be physically located in Ontario. |

**Detailed assessment for this project:**

STX is by far the most important platform for a Canadian ML basketball bettor. Key considerations:

**Strengths:**
- Only regulated exchange in Canada. Full legal protection.
- CAD deposits/withdrawals via Interac (fast, familiar for Canadians).
- Has an API for programmatic trading. Limit orders enable algorithmic market making.
- NCAA basketball markets available (spreads, moneylines for major games).
- No account limiting for winning (exchange model -- they profit from commission regardless).
- 3% commission is reasonable (lower than Betfair's standard 5%).

**Weaknesses:**
- **Ontario only.** If you are in BC, Alberta, Quebec, or elsewhere, STX is not an option.
- **Liquidity is thin.** As a relatively new exchange in a single province, many markets have limited depth. NBA is strongest; NCAAB liquidity is weaker, especially for non-marquee games.
- **No mid-major NCAAB.** You will not find lines on Horizon League or SWAC games. Coverage is limited to major conference games, rivalry games, and tournament action.
- **Props are rare.** The CLAUDE.md strategy of targeting player props and first-half totals is difficult to execute here.
- **API is developing.** Documentation suggests it may not cover all use cases. Team is receptive to integration requests.

### 4.2 Provincial Operators (Not Exchanges)

No provincial operator offers exchange-style or P2P betting:

| Province | Platform | Type | NCAA Basketball |
|----------|----------|------|-----------------|
| Ontario | OLG Proline+ | Traditional sportsbook | Yes (limited) |
| British Columbia | PlayNow | Traditional sportsbook | Yes (limited) |
| Quebec | Mise-o-jeu (Loto-Quebec) | Traditional sportsbook | Yes (limited) |
| Manitoba/Saskatchewan | Proline+ | Traditional sportsbook | Limited |
| Alberta | PlayAlberta | Traditional sportsbook | Yes (limited). New iGaming framework launching spring/summer 2026. |

All provincial operators are traditional sportsbooks that take the other side of your bet. They will limit sharp bettors. They are not suitable for algorithmic betting at scale.

### 4.3 Alberta iGaming Launch (2026)

Alberta passed the iGaming Alberta Act (Bill 48) in May 2025 and is preparing to launch a competitive iGaming market in spring/summer 2026:

- The Alberta iGaming Corporation is being established to manage commercial relationships.
- AGLC (Alberta Gaming, Liquor and Cannabis) continues as the integrity authority.
- The framework mirrors Ontario's model: private operators can apply for licenses alongside the government platform.
- BetMGM and BetRivers have already begun pre-registration.
- **No betting exchange has been announced for Alberta yet**, but the framework would theoretically allow one. STX could potentially expand to Alberta once the market opens.

**This is worth monitoring closely.** If Alberta licenses a betting exchange, it would double the Canadian exchange-accessible population.

### 4.4 Ontario Court of Appeal Ruling (2025)

A significant legal development: Ontario's Court of Appeal ruled that allowing Ontario-based online gamblers to participate in peer-to-peer games with players outside of Canada is legal. This could have implications for exchange liquidity if platforms like STX can match Ontario bettors against international counterparties.

### 4.5 Canadian Betting Exchange Startups

No other Canadian startups building betting exchanges were identified in this research beyond STX. The regulatory complexity (provincial jurisdiction, no federal framework for exchanges) creates high barriers to entry.

---

## 5. Betfair Access via Broker Services

Since Betfair is blocked in Canada, the only route to Betfair's liquidity pool is through white-label broker services. These are intermediaries that provide Betfair access in countries where Betfair itself does not operate.

### 5.1 Orbit Exchange (Betfair White Label)

| Attribute | Detail |
|-----------|--------|
| **What it is** | Betfair white-label. Accesses full Betfair liquidity pool. |
| **Canada access** | **Unclear.** Not directly accessible -- must go through a betting broker/agent. Some sources suggest Canada may be restricted even through brokers. |
| **Commission** | 3% on net winnings |
| **How to access** | Through betting brokers like AsianConnect88 or BFB247 |

### 5.2 AsianConnect88

| Attribute | Detail |
|-----------|--------|
| **What it is** | Betting broker providing access to multiple sportsbooks and exchanges |
| **Canada access** | Appears to accept Canadian players (welcome bonuses advertised for Canadian users) |
| **Betfair access** | **No longer available.** AsianConnect reportedly stopped offering Betfair and has no plans to resume. |
| **Alternatives offered** | Piwi247 (includes PIWIXchange, a Betfair white-label with 2.5% commission), Pinnacle white-label, and others |
| **Regulation** | Licensed under Government of Netherlands Antilles |

### 5.3 Assessment of Broker Route

**Pros:**
- Potential access to Betfair liquidity (the deepest exchange market in the world)
- Some brokers may offer lower commission than Betfair direct
- Can access multiple books through one account

**Cons:**
- **Counterparty risk.** Brokers are typically offshore-licensed with minimal regulatory oversight.
- **Added complexity.** Extra layer between you and your funds.
- **Uncertain Canada access.** Availability changes and may depend on specific broker policies.
- **Deposit/withdrawal friction.** May not support Interac or CAD directly.
- **API access uncertain.** Even if Betfair API-NG is technically available through the white label, broker implementation varies.
- **Not recommended for large bankrolls** unless you thoroughly vet the broker's track record and financial stability.

---

## 6. Sharp-Friendly Traditional Sportsbooks (Fallback)

While not P2P exchanges, these platforms are important fallbacks for a Canadian sharp/algorithmic bettor because they do not aggressively limit winning accounts.

### 6.1 Pinnacle -- THE SHARP-FRIENDLY STANDARD

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | **Yes.** Fully available to Canadian bettors. |
| **Account limiting** | **"Winners Welcome" policy.** Does not ban or limit winning bettors, even consistent winners at maximum stakes. Explicitly allows arbitrage betting. |
| **NCAA basketball** | Yes. All major games throughout the season. |
| **NBA** | Full coverage with high limits (up to $1,000,000) |
| **Margins** | Among the lowest in the industry (~2-3% on major markets) |
| **API** | Previously available. **Closed to general public since July 2025.** Bespoke data services available for high-value bettors and commercial partnerships. Apply at api@pinnacle.com. |
| **Deposit/Withdrawal** | Supports Canadian banking methods |

**Assessment:** Pinnacle is not an exchange (you bet against Pinnacle, not other players), but its winners-welcome policy and low margins make it the most practical alternative for a Canadian sharp bettor who cannot access true exchanges. The API closure in July 2025 is a significant setback for algorithmic betting -- you would need to negotiate bespoke access or use alternative data feeds (e.g., Betstamp Pro for odds data, then place bets manually or via scraping).

### 6.2 bet365 (Ontario)

| Attribute | Detail |
|-----------|--------|
| **Available in Canada?** | Yes. Licensed in Ontario through iGaming Ontario. Also available to non-Ontario Canadians through offshore operation. |
| **Account limiting** | **Will limit and restrict winning accounts.** This is well-documented. Once restricted, no appeal process. |
| **NCAA basketball** | Yes |
| **API** | No public API |

**Not recommended for sharp/algorithmic betting.** Account limitation is inevitable once you demonstrate consistent profitability.

### 6.3 Other Ontario-Licensed Books (FanDuel, DraftKings, BetMGM, etc.)

All Ontario-licensed traditional sportsbooks will limit sharp accounts. They are useful for:
- Comparing odds to find CLV
- Occasional soft-line exploitation before getting limited
- Data collection (scraping odds for model training)

They are NOT suitable as primary long-term betting platforms for an ML-driven strategy.

---

## 7. Platform Comparison Matrix

### For a Canadian Basketball ML Bettor

| Platform | Type | Canada Access | NCAAB Markets | Liquidity (NCAAB) | API | Commission | CAD? | Legal Status |
|----------|------|--------------|---------------|-------------------|-----|------------|------|-------------|
| **STX** | Exchange | Ontario only | Major games | Low-Medium | Yes (developing) | 3% net winnings | Yes | Fully regulated |
| **Pinnacle** | Sportsbook (sharp) | All Canada | Comprehensive | High | Closed (bespoke only) | ~2-3% margin | Varies | Grey (offshore) / Ontario licensed |
| **SX Bet** | Crypto exchange | Likely yes | Some major | Low | Yes (REST + on-chain) | 2-4% net winnings | No (USDC) | Unregulated |
| **Overtime Markets** | Crypto AMM | Yes | Some | Medium (AMM) | On-chain | 2-3% spread | No (USDC) | Unregulated |
| **Polymarket** | Prediction market | Not Ontario; grey elsewhere | Good (853 NCAA markets) | Medium | On-chain (CLOB) | ~0% | No (USDC) | Banned Ontario; grey elsewhere |
| **Azuro** | Crypto protocol | Yes | Limited | Low-Medium | On-chain | 3-5% spread | No | Unregulated |
| **Betfair (via broker)** | Exchange (indirect) | Uncertain | Major games only | High (Betfair pool) | Uncertain | 2.5-5% | No | Grey |
| **bet365** | Sportsbook | Yes | Good | High | No | ~4.5% margin | Yes | Regulated (Ontario) |

### Key Trade-offs

| Priority | Best Option |
|----------|------------|
| Legal safety | STX (Ontario) or Pinnacle |
| Lowest fees | Polymarket (~0%) or Novig (~1%, US only) |
| Best NCAAB coverage | Pinnacle or Polymarket |
| Best API for algo betting | SX Bet or STX |
| Best liquidity | Pinnacle (sportsbook) or Betfair via broker |
| CAD support | STX or Pinnacle |
| No account limits | STX, SX Bet, Overtime, Polymarket (all exchanges/crypto) |

---

## 8. Practical Recommendations for a Canadian ML Bettor

### If You Are in Ontario

**Primary platform: STX**
- Open an account, fund with Interac.
- Use the API for programmatic order placement.
- Focus on NBA and major NCAAB games where liquidity exists.
- Place limit orders at your model's price and wait for fills.
- Commission is 3% on winning bets -- factor this into your Kelly sizing.

**Secondary: Pinnacle**
- Use for NCAAB games that STX does not cover (mid-majors, smaller conferences).
- Pinnacle will not limit you for winning.
- Apply for API access at api@pinnacle.com (explain your use case -- algorithmic bettor with funded account).
- If API access is denied, use Betstamp or similar for odds data and place bets through the website.

**Supplementary: Crypto platforms**
- SX Bet for additional exchange liquidity on major basketball games.
- Polymarket is banned in Ontario -- do not use.
- Overtime Markets for AMM liquidity when order book exchanges are thin.

### If You Are Outside Ontario (BC, Alberta, Quebec, etc.)

**Primary: Pinnacle**
- Your best option for sharp basketball betting. Low margins, no account limits, comprehensive NCAAB coverage.
- Downside: You are betting against Pinnacle (house), not P2P. But their low margins (2-3%) approximate exchange economics.

**Secondary: Crypto platforms**
- SX Bet (order book exchange -- closest to true P2P)
- Overtime Markets (AMM -- always has liquidity)
- Polymarket (outside Ontario, technically accessible; best NCAAB coverage of any crypto platform with 853 NCAA markets)

**Watch: Alberta iGaming launch (spring/summer 2026)**
- If Alberta opens and licenses a betting exchange (possibly STX expansion), this dramatically improves the situation for Alberta residents.

### Operational Setup for Algorithmic Betting

```
Data Pipeline:
  Barttorvik/ESPN/KenPom --> Feature engineering --> Model predictions

Odds Comparison:
  The Odds API --> Compare model line vs. market line --> Identify +EV bets

Execution (Ontario):
  STX API --> Place limit orders at model price
  Pinnacle (manual/scrape) --> Take closing line value on NCAAB

Execution (Outside Ontario):
  Pinnacle (manual/API if approved) --> Primary execution
  SX Bet API --> Secondary for major games
  Overtime Markets (on-chain) --> Supplementary

Tracking:
  Log every bet: platform, market, model line, market line, stake, result
  Calculate CLV against Pinnacle closing lines
```

### Bankroll Allocation Suggestion

For a Canadian bettor splitting across platforms:

| Platform | Allocation | Rationale |
|----------|-----------|-----------|
| STX (Ontario) or Pinnacle | 60-70% | Regulated, lowest counterparty risk |
| SX Bet | 15-20% | Best crypto exchange for basketball |
| Overtime / Polymarket | 10-15% | Supplementary liquidity |

Keep crypto platform exposure limited due to counterparty risk (offshore licensing, smart contract risk, regulatory uncertainty).

### Tax Implications for Canadians

- **Recreational gambling winnings are generally not taxable in Canada** (unlike the US).
- **However**, if the CRA determines you are betting as a business (systematic, algorithmic, profit-seeking), winnings may be classified as business income and taxed accordingly.
- An ML-driven betting operation with API integration and systematic execution could be characterized as business income by the CRA.
- **Consult a Canadian tax professional** familiar with gambling income before scaling up.
- Crypto platform winnings add complexity: crypto-to-CAD conversions may trigger capital gains events.
- Keep meticulous records regardless of tax treatment.

---

## 9. Sources

- [Betfair Legal Countries (Caan Berry)](https://caanberry.com/betfair-legal-countries/)
- [Betfair Country Restrictions (Cheeky Punter)](https://www.cheekypunter.com/faq/betfair-country-restrictions/)
- [Betfair Registration from Canada](https://www.registration-betting-exchange.com/ca/)
- [Betfair Canada Alternatives (MyBettingSites)](https://mybettingsites.com/ca/articles/betfair-canada-alternatives)
- [Best Betting Exchange for Canadians (SBR Forum)](https://www.sportsbookreview.com/forum/sportsbooks-industry/3497039-whats-best-betting-exchange-canadians.html)
- [STX Sportsbook Review 2026 (Dimers)](https://www.dimers.com/betting/ca/ontario/stx)
- [STX Sportsbook Review 2026 (Next.io)](https://next.io/betting-sites-on/stx/)
- [STX Review 2026 (Deadspin)](https://deadspin.com/betting-canada/reviews/stx/)
- [STX Approved to Launch in Ontario (Canadian Gaming Business)](https://www.canadiangamingbusiness.com/2023/04/18/stx-ontario-betting-exchange/)
- [STX Trading API Documentation](https://wiki.stxapp.io/en/trading-api)
- [SX Bet API Documentation](https://api.docs.sx.bet/)
- [SX Bet Terms and Conditions](https://help.sx.bet/en/articles/3613372-terms-and-conditions)
- [SX Bet Review (FreeTips)](https://www.freetips.com/bookmakers/sx-bet/)
- [Overtime Markets Documentation](https://docs.overtimemarkets.xyz/)
- [Overtime Markets Analysis (BookieBuzz)](https://bookiebuzz.com/reports/overtime-markets-defi-revolution-analysis/)
- [Azuro Protocol Guide (BeInCrypto)](https://beincrypto.com/learn/azuro-guide/)
- [Azuro Ecosystem Report (DappRadar)](https://dappradar.com/blog/azuro-protocol-ecosystem-report)
- [Polymarket NCAAB Markets](https://polymarket.com/sports/cbb/games)
- [Polymarket Ontario Ban & OSC Settlement](https://www.homesfound.ca/blog/polymarket-ontario-2025-permanent-ban-osc-settlement-guide/)
- [OSC Polymarket Settlement (Official)](https://www.osc.ca/en/news-events/news/osc-reaches-settlement-current-and-former-operators-polymarket-breach-binary-options-ban)
- [Polymarket Geo-blocking Documentation](https://docs.polymarket.com/polymarket-learn/FAQ/geoblocking)
- [Polymarket Supported/Restricted Countries (DataWallet)](https://www.datawallet.com/crypto/polymarket-restricted-countries)
- [Will Prediction Markets Become Legal in Canada? (Casino.org)](https://www.casino.org/news/prediction-markets-legality-canada/)
- [Should Prediction Markets Be Legal in Canada? (Canadian Affairs)](https://www.canadianaffairs.news/2025/08/08/polymarket-legal-in-canada-or-not/)
- [Polymarket Taxation in Canada (Tax Partners)](https://www.taxpartners.ca/polymarket-in-canada-navigating-legalities-taxation-and-record-keeping-for-prediction-markets)
- [Why Canadians Can't Trade on Kalshi (Medium)](https://robtyrie.medium.com/the-invisible-wall-why-canadians-cant-trade-predictions-on-platforms-like-kalshi-79e8f818b7b3)
- [Prediction Markets Head into Basketball Season (CNBC)](https://www.cnbc.com/amp/2026/02/11/prediction-markets-head-into-basketball-season-after-super-bowl-high-from-super-bowl.html)
- [Sporttrade CFTC Registration (SBC Americas)](https://sbcamericas.com/2026/02/04/sporttrade-applies-cftc-registration/)
- [Novig CFTC Registration (SBC Americas)](https://sbcamericas.com/2026/01/27/report-novig-cftc-registration/)
- [ProphetX and Novig Sweepstakes Pivot (SBC Americas)](https://sbcamericas.com/2024/09/05/prophetx-novig-sweepstakes/)
- [Pinnacle Sportsbook Review 2026 (OddsShark)](https://www.oddsshark.com/sportsbook-review/pinnacle-sportsbook)
- [Is Pinnacle Legal in Canada?](https://www.pinnacleoddsdropper.com/blog/is-pinnacle-legal-in-canada)
- [Pinnacle API Documentation (GitHub)](https://github.com/pinnacleapi/pinnacleapi-documentation)
- [Pinnacle Betting Limits (SureBetMonitor)](https://surebetmonitor.com/knowledge-base/pinnacle-sports-betting-limits/)
- [Orbit Exchange Review (Caan Berry)](https://caanberry.com/orbit-exchange-review/)
- [Betfair White-Label Comparison (Arbusers)](https://arbusers.com/orbitexch-vs-sharpexch-vs-fairexchange-vs-piwi247-comparing-the-top-betfair-white-label-platforms-t10254/)
- [AsianConnect88 Canada Review](https://canada-betting.com/asianconnect88/)
- [Betfair Broker for Canadians (SBR Forum)](https://sportsbookreview.com/forum/sportsbooks-industry/3271004-betfair-broker-canadians.html)
- [Bet Brokers Guide 2026 (Global Extra Money)](https://globalextramoney.com/bookmakers/bet-broker)
- [Alberta iGaming Framework 2026 (SCCG Management)](https://sccgmanagement.com/sccg-articles/2025/12/4/how-the-alberta-2026-igaming-framework-could-rewire-north-americas-online-betting-future/)
- [Alberta iGaming Launch (iGaming Business)](https://igamingbusiness.com/gaming/framework-alberta-igaming-sports-betting-online-casino/)
- [Alberta iGaming Spring/Summer 2026 (Casino.org)](https://www.casino.org/news/canadian-gaming-alberta-spring-summer-market-launch/)
- [BetRivers Alberta Pre-Registration (SBR)](https://www.sportsbookreview.com/news/betrivers-advances-alberta-plans-with-pre-registration-feb-11-2026/)
- [Bet365 Account Limiting (Caan Berry)](https://caanberry.com/bet365-account-limited/)
- [Canada Sports Betting Legal Guide (OddsShark)](https://www.oddsshark.com/canada)
- [Ontario Sports Betting Sites 2026 (SI)](https://www.si.com/betting/canada/ontario)
