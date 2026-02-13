# Canadian Tax & Legal Considerations for Algorithmic Sports Betting

*Research compiled February 2026*

**DISCLAIMER:** This is research, not legal or tax advice. Consult a Canadian tax lawyer and CPA before making decisions based on this information. Tax law is fact-specific and evolving.

---

## Table of Contents

1. [Tax Treatment of Gambling Winnings](#1-tax-treatment-of-gambling-winnings)
2. [Professional Gambler Status](#2-professional-gambler-status)
3. [The ML/Algorithm Problem: When Does a System Make You a "Business"?](#3-the-mlalgorithm-problem)
4. [Legal Considerations for Algorithmic Betting](#4-legal-considerations-for-algorithmic-betting)
5. [Currency and Banking](#5-currency-and-banking)
6. [Record Keeping](#6-record-keeping)
7. [Provincial Variations](#7-provincial-variations)
8. [Strategies to Manage Tax Risk](#8-strategies-to-manage-tax-risk)
9. [Key Case Law Summary Table](#9-key-case-law-summary-table)
10. [Sources](#10-sources)

---

## 1. Tax Treatment of Gambling Winnings

### The General Rule: Winnings Are Tax-Free

Canada's position is fundamentally different from the United States. For recreational gamblers, gambling winnings are **not taxable income** under the Income Tax Act. The CRA treats casual gambling winnings as a **windfall** -- an unexpected gain that falls outside the definition of income.

Key points:
- No reporting requirement for recreational gambling winnings
- No withholding tax on Canadian gambling winnings (unlike the US 24-30% withholding)
- Gambling losses are **not deductible** for recreational gamblers (the mirror of the non-taxable treatment)
- Investment income earned *on* gambling winnings (interest, dividends, capital gains) IS taxable

This applies to all forms of gambling: casino games, lotteries, sports betting, poker, and online gambling.

### When Gambling Winnings Become Taxable

Gambling winnings become taxable as **business income** when the CRA determines you are "carrying on the business of gambling" under the Income Tax Act. The CRA's authoritative guidance is found in **Income Tax Folio S3-F9-C1** (paragraphs 1.11-1.15).

The CRA acknowledges a critical nuance (para 1.12): **"gambling is always undertaken in pursuit of profit"** -- everyone who gambles wants to win money. Therefore, mere profit motive is NOT sufficient to establish a business. Something more is required.

The landmark quote from **Leblanc v. The Queen** (cited in para 1.13):

> "Gambling -- even regular, frequent and systematic gambling -- is something that by its nature is not generally regarded as a commercial activity except under very exceptional circumstances."

This is a high bar. The CRA must demonstrate "very exceptional circumstances" to classify gambling as business income.

### The Four CRA Factors (Income Tax Folio S3-F9-C1, para 1.15)

The CRA examines four primary criteria:

| Factor | What CRA Looks For |
|--------|-------------------|
| **1. Degree of organization** | Business-like structure: record-keeping, bankroll management, dedicated workspace, business planning |
| **2. Special knowledge or inside information** | Expertise that reduces the element of chance -- statistical models, algorithms, insider information |
| **3. Profit intent vs. pleasure** | Is gambling your livelihood or entertainment? Sole/primary income source vs. side hobby |
| **4. Extent of activities** | Frequency, volume, and regularity of bets; time devoted to the activity |

**No single factor is determinative.** The CRA and courts apply a holistic assessment of all circumstances.

### If Classified as Business Income: Tax Rates and Deductions

If your gambling is classified as business income, the consequences are significant:

**Federal tax rates (2026):**
- First $58,523: 14%
- $58,523 - $117,045: 20.5%
- $117,045 - $161,733: 26%
- $161,733 - $230,451: 29%
- Over $230,451: 33%

**Provincial tax adds 5% to 25.75%** depending on province. Combined marginal rates range from ~20% (low income) to ~53.5% (top bracket in some provinces like Ontario).

**Self-employment CPP contributions (2026):** 11.9% on income up to $74,600, plus additional 8% on income between $74,600 and $85,000.

**Deductible expenses** (the silver lining of business classification):
- Gambling losses (up to the amount of winnings; cannot create a net loss against other income)
- Software and analytical tools (including cloud computing costs for ML training)
- Data subscriptions (KenPom, The Odds API, etc.)
- Computer hardware and equipment
- Internet and home office expenses (proportional)
- Professional fees (accountant, tax advisor)
- Travel expenses related to gambling
- Educational materials and courses
- Platform/exchange commissions and fees

**Filing deadline:** June 15 for self-employed (but taxes owed must be paid by April 30).

---

## 2. Professional Gambler Status

### How CRA Defines a Professional Gambler

There is no formal "professional gambler" registration or status in Canada. The determination is made retroactively by the CRA during an assessment or audit, based on the totality of circumstances. The legal framework comes from the Supreme Court of Canada's decision in **Stewart v. Canada** (2002 SCC 46), which established the test:

> "Does the taxpayer intend to carry on an activity for the purpose of making a profit and is there evidence to support this intention?"

This combines **subjective intent** (what you intended) with **objective evidence of commerciality** (what it looks like from the outside).

### Factors CRA Considers (Drawn from Case Law)

From **Stewart**, **Moldowan v. The Queen**, and subsequent gambling cases, courts examine:

1. **Prior profit/loss history** -- Consistent profitability over multiple years
2. **Training and preparation** -- Study of games, strategy development, skill improvement
3. **Time commitment** -- Hours devoted to gambling and related analysis
4. **Capacity to generate profit** -- Demonstrable skill advantage over other players/the market
5. **Risk management strategies** -- Bankroll management, Kelly criterion, position sizing
6. **Business planning and structure** -- Systematic approach, dedicated workspace, separate accounts
7. **Use of technology and tools** -- Software for tracking, analysis, statistical modeling
8. **Income dependency** -- Whether gambling is the sole or primary income source
9. **Record-keeping** -- Detailed tracking of bets, wins, losses, and analysis
10. **Third-party arrangements** -- Staking agreements, profit-sharing, coaching relationships

### Comparison with US Treatment

| Aspect | Canada | United States |
|--------|--------|--------------|
| Recreational winnings | **Tax-free** | **Fully taxable** (reported on Form W-2G / 1040) |
| Recreational losses | Not deductible | Deductible up to winnings (itemizers only) |
| Professional winnings | Taxable as business income | Taxable as business income (Schedule C) |
| Professional losses | Deductible against gambling income | Fully deductible against gambling income |
| Withholding | None (Canadian sources) | 24% federal withholding on certain thresholds |
| Self-employment tax | CPP contributions (~12%) | 15.3% FICA (Social Security + Medicare) |

**The Canadian advantage is enormous for recreational bettors.** A US bettor paying 24% federal + state tax on all winnings faces a massive structural disadvantage compared to a Canadian recreational bettor paying 0%.

---

## 3. The ML/Algorithm Problem

### Critical Question: Does Using an ML Model Make You a "Business"?

This is the most important question for this project. The answer is: **it significantly increases the risk, but it is not automatically determinative.**

### Arguments That ML Betting IS a Business (Risk Factors)

The following characteristics of an ML-based betting system map directly onto the CRA's four factors:

**Factor 2 (Special Knowledge):** An ML model trained on historical data, generating probability estimates, and identifying market inefficiencies is textbook "special knowledge that reduces the element of chance." This is arguably the strongest risk factor. The CRA's Folio explicitly says that using expertise to reduce chance is a key indicator of business activity.

**Factor 1 (Organization):** Building data pipelines, training models, backtesting strategies, maintaining code repositories, and systematically executing bets based on model output demonstrates a high degree of organization.

**Factor 4 (Extent):** If the model is placing hundreds of bets per season across multiple markets, this demonstrates high frequency and volume.

**Factor 3 (Profit Intent):** The entire premise of the project is to find "profitable edges" -- this is explicitly a profit-seeking enterprise, not entertainment.

### Arguments That ML Betting is NOT a Business

**The Leblanc Defense:** Even the Leblanc brothers, who wagered $10-13 million per year on sports lotteries using a computer program to analyze bets and earned $5 million profit, were found NOT to be running a business. The court found their success was "pure luck, not a result of organized or intentional efforts" despite the apparent systematization.

**The "Pizza and Xbox" Defense:** In a related case, brothers betting $200,000-$300,000 weekly on sports were found to be recreational because their approach lacked true systematic organization despite the massive volume.

**The CRA's Own Standard:** "Gambling -- even regular, frequent and systematic gambling -- is something that by its nature is not generally regarded as a commercial activity except under very exceptional circumstances." This sets a HIGH bar.

**Key Distinction -- Sports Betting vs. Poker:** The cases that have most aggressively found business income (Fournier-Giguere 2025 FCA 112, D'Auteuil, Berube) involved **professional poker players** where:
- Poker was their SOLE income source
- They devoted nearly ALL their waking hours to poker
- They earned millions ($1.4M - $5.2M over assessed years)
- They used third-party software to analyze opponents in real-time
- They could "reasonably expect to make a living by playing poker"

Sports betting with an ML model is different from full-time professional poker in important ways -- it can be done alongside regular employment, it requires less daily time commitment, and the model does most of the "work."

### Risk Assessment for This Project

**Higher risk indicators (avoid these if possible):**
- Quitting your day job to bet full-time
- Gambling being your primary income source
- Betting hundreds of thousands of dollars per year
- Maintaining extensive formal business infrastructure
- Sharing your system with others for a fee
- Using staking arrangements or investors

**Lower risk indicators (maintain these):**
- Having significant non-gambling income (employment, investments)
- Treating betting as a hobby alongside a career
- Keeping bet sizes modest relative to total income
- Not advertising yourself as a professional bettor
- Not selling picks or model access
- Keeping the activity part-time

### The Paradox

There is a well-recognized paradox in Canadian gambling tax law, noted by judges themselves: the **post hoc ergo propter hoc** fallacy. If you win consistently, the CRA argues you must have had a system (therefore business). If you lose consistently, the CRA ignores you. This creates an asymmetry where successful systematic bettors face taxation while unsuccessful ones face nothing -- "heads I tax you, tails I ignore you."

The Duhamel case (2022 TCC) pushed back on this, finding that a World Series of Poker winner's winnings were NOT taxable business income despite his obvious skill, because his activities did not display sufficient "commerciality" and business-like conduct beyond simply being good at poker.

---

## 4. Legal Considerations for Algorithmic Betting

### Is Using ML Models for Betting Legal in Canada?

**Yes.** There is no Canadian law prohibiting the use of algorithms, statistical models, machine learning, or any analytical tool for making betting decisions. Using superior analysis to inform betting decisions is not a criminal offense under the Criminal Code.

What IS illegal under the Criminal Code (Section 202-209):
- Operating an unlicensed gambling business
- Cheating at play (e.g., manipulating outcomes, match-fixing)
- Bookmaking without a license

Using a model to inform your own bets as a bettor is not covered by any of these provisions.

### Bot/Automation Policies

While using ML for decision-making is legal, **automated execution** (bots that place bets programmatically) is a different matter governed by platform terms of service, not criminal law.

**Ontario-licensed platforms (iGaming Ontario market):**
- Most licensed sportsbooks explicitly prohibit automated betting in their terms of service
- The AGCO (Alcohol and Gaming Commission of Ontario) requires operators to detect and prevent suspicious activity, which may include bot detection
- Violations of ToS are civil matters (account closure, fund seizure), not criminal

**STX Exchange (Ontario's only legal betting exchange):**
- Ontario's first and only peer-to-peer betting exchange, licensed by AGCO
- ToS should be reviewed carefully for API/automation policies
- Exchange models are inherently more tolerant of sophisticated bettors than traditional sportsbooks

**Offshore/crypto platforms:**
- Generally more tolerant of automated betting
- Some explicitly offer APIs (SX Bet, Betfair in non-US markets)
- Legal risk: using unlicensed offshore platforms is itself a gray area in Canada

**Practical approach:** Use your ML model to generate picks and sizing recommendations, then execute bets manually. This sidesteps all ToS bot concerns while capturing nearly all the model's value.

### Data Scraping Legality in Canada

Data scraping for training your ML model exists in a complex legal landscape as of 2026:

**Copyright considerations:**
- Scraping copyrighted content (articles, analysis, commentary) without permission violates the Copyright Act
- Scraping publicly available factual data (box scores, statistics, odds) is generally safer since facts are not copyrightable
- The November 2024 lawsuit by Canadian news companies against OpenAI established that scraping copyrighted content at scale can trigger copyright infringement claims

**Terms of Service:**
- Violating a website's ToS by scraping is a civil matter, not criminal
- CanLII v. Caseway AI (November 2024, BC Supreme Court) established that ToS-prohibited scraping can support legal claims
- Check robots.txt and ToS of any site you scrape

**Privacy considerations:**
- PIPEDA (federal) and provincial privacy laws apply to personal information
- The Clearview AI Alberta decision (May 2025) provided some legitimization of scraping publicly available, non-password-protected data, with the court finding potential Charter protection for automated data collection
- Sports statistics generally do not contain personal information that would engage PIPEDA

**Practical guidance for this project:**
- Scraping box scores, statistics, and game results from public sources: **low risk**
- Using established APIs (ESPN, sports reference sites): **low risk** if respecting rate limits
- Scraping odds data: **low risk** for publicly posted odds; check ToS
- Scraping proprietary analysis/ratings (e.g., KenPom without subscription): **higher risk**, pay for subscriptions instead
- Use `cbbpy`, `kenpompy` (with subscription), and public APIs as recommended in CLAUDE.md

### Peer-to-Peer Betting Platforms Available in Canada

| Platform | Available in Canada? | Notes |
|----------|---------------------|-------|
| **STX** | Yes (Ontario) | Only legal P2P exchange in Canada, AGCO-licensed |
| **Betfair Exchange** | No | Withdrew from Canada; blocks Canadian IPs; VPN use violates ToS |
| **Sporttrade** | No | US-only (NJ, CO, AZ, IA, VA) |
| **Novig** | No | US-only (45 states); applying for CFTC DCM status |
| **Prophet Exchange** | No | US-only (NJ) |
| **Polymarket** | No | Banned by Ontario securities regulator (2025); CSA binary options ban |
| **Kalshi** | No | US CFTC-regulated, not available in Canada |
| **SX Bet** | Gray area | Blockchain-based, no Canadian regulation, accessible but not licensed |
| **Overtime Markets** | Gray area | DeFi protocol, accessible but not licensed |
| **Azuro** | Gray area | DeFi protocol, accessible but not licensed |

**The Canadian P2P landscape is extremely limited.** STX in Ontario is the only legal option. Crypto/DeFi platforms are accessible but operate in a regulatory gray zone.

---

## 5. Currency and Banking

### CAD vs. USD Considerations

Most sports betting markets, odds data, and sophisticated platforms are USD-denominated. This creates several considerations:

**Foreign exchange costs:**
- Converting CAD to USD and back incurs FX spreads (typically 1.5-3% at banks, 0.5-1.5% at dedicated FX services like Wise or Norbert's Gambit via a brokerage)
- For frequent transactions, these costs eat into already-thin betting edges
- Consider maintaining a USD account to minimize conversions

**Capital gains on FX:**
- If you hold USD and it appreciates against CAD, the gain is technically a capital gain
- If gambling is classified as a business, FX gains/losses are part of business income
- Track your cost basis for all USD holdings

### Banking Considerations

**Bank account treatment:**
- Canadian banks can and do close accounts associated with gambling activity
- Large or frequent deposits/withdrawals from gambling sites may trigger FINTRAC reporting
- FINTRAC requires casinos to report disbursements of $10,000+ CAD (single or aggregate in 24 hours)
- Banks must report suspicious transactions; gambling-related patterns can be flagged

**Practical banking strategies:**
- Keep gambling funds in a separate account from primary banking
- Use a bank or credit union that is known to be tolerant of gambling transactions
- Avoid structuring deposits/withdrawals to stay below $10,000 (this is itself suspicious under FINTRAC guidelines)
- Consider Interac e-Transfer for deposits to Ontario-licensed platforms (widely supported, no intermediary visibility)

### Crypto On/Off Ramps

For decentralized betting platforms:
- Canadian crypto exchanges (Shakepay, Newton, Bitbuy, Kraken Canada) allow CAD-to-crypto conversion
- KYC is required on all Canadian-regulated crypto exchanges
- Converting crypto gambling winnings to fiat may trigger **capital gains** on the crypto appreciation, regardless of whether the gambling winnings themselves are taxable
- Using crypto to place bets is a **disposition** that may trigger capital gains/losses on the crypto itself
- Stablecoins (USDT, USDC) minimize crypto volatility risk but still require tracking cost basis

---

## 6. Record Keeping

### What Records to Keep

Regardless of whether you believe your gambling is recreational or business, maintain thorough records. If the CRA ever questions your gambling income, documentation is your best defense -- either proving you are recreational OR supporting your deductions if classified as business.

**Essential records (minimum 6 years, though 5 years is the legal minimum):**

**Betting records:**
- Date and time of every bet
- Platform/exchange used
- Sport, league, game, and market (spread, total, moneyline, etc.)
- Odds at time of bet
- Stake amount
- Result (win/loss/push)
- Net profit/loss per bet
- Running balance

**Financial records:**
- All deposits to and withdrawals from betting platforms
- Bank statements showing gambling transactions
- Currency conversion records (CAD/USD) with exchange rates
- Platform statements and transaction histories
- Screenshots of account balances (platforms can disappear)

**Model/analysis records (if you choose to keep them -- see risk note below):**
- Model predictions vs. actual outcomes
- Feature importance and model performance metrics
- Backtesting results

**Expense records (if claiming business income):**
- Receipts for all software, subscriptions, and tools
- Data subscription invoices
- Computer hardware purchases
- Internet bills (proportional allocation)
- Home office measurements and calculations
- Professional fees (accountant, lawyer)

### Documenting the Recreational vs. Professional Distinction

**If you want to maintain recreational status, document evidence supporting that:**
- You have substantial non-gambling income (employment records, T4 slips)
- Gambling is a part-time hobby, not your livelihood
- You do not advertise gambling services or sell picks
- Gambling does not dominate your time (work schedules, vacation records)
- You treat it as entertainment with an analytical twist

**Risk note on model documentation:** Detailed records of your ML system, backtesting, and systematic approach could be used *against* you by the CRA to argue business status. This creates a tension: you want records for good bankroll management, but detailed records also evidence "organization" and "special knowledge." Consider keeping betting records separate from model development records, and consult a tax lawyer about what to retain vs. what exposure documentation creates.

---

## 7. Provincial Variations

### Federal vs. Provincial Jurisdiction

Income tax classification (recreational vs. business) is a **federal** matter under the Income Tax Act. There are no provincial variations in how gambling income is classified. However, provinces differ in:

**Tax rates (combined federal + provincial top marginal rate, 2026):**

| Province | Top Marginal Rate | Notes |
|----------|------------------|-------|
| Ontario | ~53.5% | Largest regulated iGaming market |
| Quebec | ~53.3% | Slightly lower than Ontario |
| British Columbia | ~53.5% | Similar to Ontario |
| Alberta | ~48% | No provincial sales tax; lowest top rate among major provinces |
| Manitoba | ~50.4% | Mid-range |
| Saskatchewan | ~47.5% | Lower than most |

### Provincial Gambling Regulation Differences

**Ontario:**
- Most mature regulated iGaming market (48+ licensed operators as of 2026)
- Only province with a legal P2P betting exchange (STX)
- iGaming Ontario (iGO) oversees the private market
- 20% operator revenue share
- Minimum gambling age: 19

**Quebec:**
- Loto-Quebec operates Mise-o-jeu+ (provincial monopoly model)
- No private iGaming market (yet -- industry is pushing for Ontario-style regulation)
- Estimated $300M+ annual revenue lost to unlicensed sites
- Minimum gambling age: 18
- Has attempted (controversially) to mandate ISP blocking of unlicensed gambling sites

**Alberta:**
- iGaming Alberta Act (Bill 48) passed March 2025
- Private-sector licensed market expected to launch early 2026
- Currently only Play Alberta (provincial platform) is legal
- Minimum gambling age: 18
- Once launched, will likely resemble Ontario's model

**British Columbia:**
- PlayNow.com (BCLC) is the only legal online platform
- No private iGaming market
- BCLC has been more restrictive than Ontario
- Minimum gambling age: 19

**Other provinces:**
- Most operate through provincial lottery corporations
- Limited or no private online gambling
- Saskatchewan, Manitoba, and Atlantic provinces have their own provincial platforms

### Provincial Implications for This Project

- **If in Ontario:** Best positioning -- access to STX exchange, 48+ licensed sportsbooks, mature regulatory environment
- **If in Alberta:** Wait for iGaming market launch (expected early 2026); currently limited to Play Alberta and offshore
- **If in Quebec/BC:** Limited to provincial platforms and offshore; no legal P2P exchange
- **Regardless of province:** Federal tax treatment is the same everywhere

---

## 8. Strategies to Manage Tax Risk

### If You Want to Stay "Recreational" (Tax-Free)

1. **Maintain significant employment or business income** that clearly exceeds gambling profits
2. **Keep gambling part-time** -- do not devote more time to betting than to your primary occupation
3. **Do not depend on gambling income** for living expenses
4. **Keep bet sizes modest** relative to your overall financial picture
5. **Do not advertise yourself** as a professional bettor, tout, or model seller
6. **Avoid staking arrangements** or investors in your betting activity
7. **Do not incorporate** a gambling business or open a business account for gambling
8. **Frame the ML model as a hobby project** -- an intellectual exercise in data science that happens to inform occasional bets
9. **Maintain a diverse gambling portfolio** -- bet on things for fun too, not just model-identified edges
10. **Consult a tax lawyer proactively** to understand your specific risk profile

### If You Accept Business Classification (or Want It)

There are scenarios where business classification is actually advantageous:

- **Net losses:** If you have losing years, business losses can offset gambling income in other years (carryback 3 years, carryforward 20 years)
- **Expense deductions:** Software, data, hardware, and other costs reduce taxable income
- **CPP contributions:** Build retirement benefits (though at ~12% cost)
- **Legitimacy:** Clean reporting avoids future CRA disputes

If you go this route:
1. Report all gambling income on your T1 as business income (line 13500)
2. Deduct all legitimate expenses
3. Keep meticulous records
4. File by June 15 (self-employed deadline) but pay by April 30
5. Consider HST/GST implications if annual revenue exceeds $30,000 (though gambling is generally an exempt supply)
6. Engage a CPA experienced with gambling income

### The Middle Path

Many sophisticated Canadian bettors take this approach:
- Keep detailed records privately
- Do not report gambling winnings (treating them as recreational windfalls)
- Do not claim gambling losses or expenses
- Keep the activity modest enough to not attract CRA attention
- Have a tax lawyer on retainer who can respond if CRA ever questions the activity
- Be prepared to argue recreational status using the factors above

This is the most common approach but carries risk if the CRA disagrees with your self-classification.

---

## 9. Key Case Law Summary Table

| Case | Year | Outcome | Key Facts | Relevance |
|------|------|---------|-----------|-----------|
| **Stewart v. Canada** | 2002 SCC | Established test | Supreme Court defined business vs. hobby test | Foundation for all gambling tax cases |
| **Leblanc v. The Queen** | 2006 TCC | NOT taxable | $10-13M/year wagered on sports, $5M profit, computer program used | Even massive systematic sports betting can be recreational |
| **Luprypa v. The Queen** | TCC | Taxable | Pool player who targeted inebriated opponents | Skill reducing chance = business |
| **Duhamel v. Canada** | 2022 TCC | NOT taxable | WSOP winner; obvious poker skill | Skill alone insufficient; need commerciality |
| **Fournier-Giguere v. Canada** | 2025 FCA | Taxable | $1.45M poker income; sole income; full-time; used tracking software | Most important recent case; sets high-water mark for "business" gambling |
| **D'Auteuil v. The King** | 2023 TCC | Taxable | $5.24M poker income; main income source; full-time | Consistent profitability + full-time commitment = business |
| **Berube v. The King** | 2023 TCC | Taxable | $3.22M poker income; sole income; full-time | Same as D'Auteuil |

### Key Takeaways from Case Law

1. **Volume alone is not enough** -- The Leblanc brothers wagered millions and still were not taxed
2. **Skill alone is not enough** -- Duhamel won the WSOP and still was not taxed
3. **The combination that triggers taxation:** Full-time commitment + sole/primary income + consistent profitability + use of professional tools + business-like organization
4. **Sports betting has been treated MORE favorably** than poker in case law -- no Canadian court has yet classified sports betting as business income in a published decision of this prominence
5. **The 2025 Fournier-Giguere FCA decision** is the strongest statement yet that skilled, systematic gambling CAN be business income, but the facts were extreme (millions in income, sole livelihood, nearly all waking hours devoted to poker)

---

## 10. Sources

### Official Government Sources
- [CRA Income Tax Folio S3-F9-C1: Lottery Winnings, Miscellaneous Receipts](https://www.canada.ca/en/revenue-agency/services/tax/technical-information/income-tax/income-tax-folios-index/series-3-property-investments-savings-plans/series-3-property-investments-savings-plans-folio-9-miscellaneous-payments-receipts/income-tax-folio-s3-f9-c1-lottery-winnings-miscellaneous-receipts-income-losses-crime.html)
- [CRA: Amounts That Are Not Reported or Taxed](https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/amounts-that-taxed.html)
- [CRA: 2026 Tax Rates and Income Brackets](https://www.canada.ca/en/revenue-agency/services/tax/individuals/frequently-asked-questions-individuals/canadian-income-tax-rates-individuals-current-previous-years.html)
- [FINTRAC: Record Keeping Requirements for Casinos](https://fintrac-canafe.canada.ca/guidance-directives/recordkeeping-document/record/cas-eng)
- [FINTRAC: Special Bulletin on Online Gambling](https://fintrac-canafe.canada.ca/intel/bulletins/gambling-jeu-eng)

### Case Law and Legal Analysis
- [Fournier-Giguere v. Canada, 2025 FCA 112 -- Tax Interpretations Summary](https://taxinterpretations.com/content/994263)
- [How Are Professional Poker Winnings Taxed in Canada? (TaxPage / Mondaq)](https://www.mondaq.com/canada/income-tax/1651356/how-are-professional-poker-winnings-taxed-in-canada-taxable-business-income-and-non-taxable-hobbies-fournier-gigu%C3%A8re-v-canada-et-al-2025-fca-112)
- [FCA: Poker Can Be a Business (Canadian Tax Foundation)](https://www.ctf.ca/EN/EN/Newsletters/Canadian_Tax_Focus/2025/3/250315.aspx)
- [Texas Hold 'Em or Taxes Hold 'Em? (BLG Law)](https://www.blg.com/en/insights/2023/07/texas-hold-em-or-taxes-hold-em-taxes-and-gambling-in-canada)
- [Are Poker Winnings Now Taxable in Canada? 4 Recent Decisions (Canadian Accountant)](https://www.canadian-accountant.com/content/practice/poker-winnings-tax-court-decisions)
- [Taxation of Gambling and Poker Winnings -- Toronto Tax Lawyer Guide (TaxPage)](https://taxpage.com/articles-and-tips/gambling-poker-winnings/)
- [Are Gambling Winnings a "Prize" Under the Income Tax Act? (LawNow)](https://www.lawnow.org/are-gambling-winnings-a-prize-under-the-income-tax-act/)
- [Gambling and Taxable Implications in Canada (Rosen Tax Law)](https://rosentaxlaw.com/gambling-and-taxable-implications/)

### Tax Guidance and CPA Analysis
- [Do You Have to Pay Tax on Gambling or Sports Betting? (Lucas CPA)](https://www.lucas.cpa/blog/do-i-pay-tax-on-sports-betting-canada)
- [Online Gambling and Lottery Winnings in Canada -- Complete Guide (Mackisen CPA)](https://mackisen.com/blog/online-gambling-and-lottery-winnings-in-canada-what-s-taxable-and-what-isn-t-a-complete-guide)
- [Taxes on Lottery & Gambling Winnings in Canada (Accounting Montreal)](https://accountingmontreal.ca/accounting-insights/taxes-on-lottery-and-gambling-winnings-in-canada-what-you-need-to-know/)
- [How to Report Gambling Winnings in Canada -- 2025 Tax Guide](https://www.deucescracked.com/how-to-report-gambling-winnings-in-canada-2025-tax-guide/)
- [Gambling Winnings Tax Canada: CRA and US Withholding](https://thegamingboardroom.com/2025/09/18/gambling-winnings-tax-canada-cra-and-us-withholding/)
- [2026 Canadian Income Tax Brackets (Fidelity)](https://www.fidelity.ca/en/insights/articles/canadian-income-tax-brackets/)
- [Self-Employed Taxes in Canada (NerdWallet)](https://www.nerdwallet.com/ca/p/article/finance/about-self-employed-taxes)

### Regulatory and Legal Framework
- [Gaming Law 2025 -- Canada (Chambers and Partners)](https://practiceguides.chambers.com/practice-guides/gaming-law-2025/canada)
- [Where Sports Betting Is Legal in Canada (RG.org)](https://rg.org/en-ca/guides/regulations)
- [Canada's Legal Sports Betting Guide 2026 (BettingTop10)](https://www.bettingtop10.ca/legal-betting/)
- [Sports Betting Laws Canada 2025 -- Provincial Guide](https://motron-ltd.com/2025/06/06/understanding-sports-betting-regulations-in-canada-a-2025-update/)
- [Will Prediction Markets Ever Become Legal in Canada? (Casino.org)](https://www.casino.org/news/prediction-markets-legality-canada/)
- [STX Sportsbook Review 2026 -- Ontario Betting Exchange](https://next.io/betting-sites-on/stx/)

### Data Scraping Legal Landscape
- [Alberta Court Legitimizes Data Scraping (BLG Law)](https://www.blg.com/en/insights/2025/06/alberta-judgment-opens-the-door-to-the-legitimization-of-data-scraping-and-ai-model-training)
- [Scraping the Surface: OpenAI Sued in Canada (ABA)](https://www.americanbar.org/groups/business_law/resources/business-law-today/2025-february/openai-sued-data-scraping-canada/)
- [Legality of Data Scraping Using AI in Canada (Torkin Manes)](https://www.torkin.com/insights/publication/legality-of-data-scraping-using-ai-revisiting-in-canada)
- [Data Scraping Under Fire: Lessons for Canadian Companies (Dentons)](https://www.dentonsdata.com/data-scraping-under-fire-what-canadian-companies-can-learn-from-kasprs-e240k-fine/)

### Banking and FINTRAC
- [New FINTRAC Requirements Effective October 2025 (McCarthy Tetrault)](https://www.mccarthy.ca/en/insights/blogs/techlex/reminder-new-fintrac-requirements-effective-october-1-2025)
- [FINTRAC: Reporting Casino Disbursements](https://fintrac-canafe.canada.ca/guidance-directives/transaction-operation/cdr/casino-eng)
