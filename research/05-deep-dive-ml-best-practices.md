# Deep Dive: ML Best Practices for Basketball Betting (2024-2026)

> Compiled February 2026. This document covers practical implementation details and recent
> developments beyond the basics already covered in RESEARCH.md and CLAUDE.md.

---

## Table of Contents

1. [Model Architecture Best Practices](#1-model-architecture-best-practices)
2. [Feature Engineering Deep Dive](#2-feature-engineering-deep-dive)
3. [Validation and Evaluation](#3-validation-and-evaluation)
4. [Practical Implementation](#4-practical-implementation)
5. [Common Mistakes and How to Avoid Them](#5-common-mistakes)
6. [What's New in 2025-2026](#6-whats-new-2025-2026)
7. [Canada-Specific Considerations](#7-canada-specific)
8. [Sources](#8-sources)

---

## 1. Model Architecture Best Practices

### 1.1 The GBM vs Neural Net vs Hybrid Question

**The consensus as of 2026: Gradient Boosted Machines (GBMs) remain the best starting point
for tabular sports prediction data, but the gap with neural approaches is narrowing.**

Recent benchmarks and papers confirm:

- **XGBoost and LightGBM** remain the dominant workhorses for structured/tabular sports
  prediction. In a 2025 PLOS One systematic review of AI for basketball prediction, tree-based
  ensembles (Random Forest, XGBoost, LightGBM) were the most commonly used and consistently
  high-performing methods across studies.

- **LightGBM** offers faster training via histogram-based learning, which matters during
  daily retraining. It achieves comparable or slightly better performance to XGBoost on
  basketball datasets while training significantly faster.

- **Neural networks** show advantages only in specific scenarios:
  - When you have sequential/temporal data (game-by-game sequences) -- LSTMs shine here
  - When you want uncertainty quantification via Monte Carlo dropout
  - When you have graph-structured data (team interaction networks) -- GNNs
  - When you need to learn embeddings for categorical entities (teams, players)

- **Key finding from a January 2026 study** (Uncertainty-Aware ML for NBA Forecasting):
  Logistic regression achieved the best Brier score (0.199) and log-loss (0.583) among
  tabular baselines, while XGBoost had slightly worse calibration (Brier 0.202, log-loss
  0.589) but higher AUC. This is significant -- for betting, calibration matters more than
  discrimination.

**Practical recommendation:** Start with logistic regression as a baseline, then XGBoost/
LightGBM. Only add neural components when you have a specific reason (embeddings, sequences,
uncertainty). The marginal gain from complex architectures is small relative to the risk of
overfitting.

### 1.2 GBM Hyperparameters That Matter

For XGBoost/LightGBM on basketball data, the research converges on these settings:

```python
# XGBoost parameters for basketball prediction
xgb_params = {
    'max_depth': 3-5,           # Shallow trees to prevent overfitting
    'n_estimators': 200-500,    # More trees with lower learning rate
    'learning_rate': 0.01-0.05, # Low learning rate + more trees
    'subsample': 0.7-0.8,       # Row sampling for regularization
    'colsample_bytree': 0.7-0.8,# Column sampling
    'min_child_weight': 5-10,   # Higher for small datasets
    'reg_alpha': 0.1-1.0,       # L1 regularization
    'reg_lambda': 1.0-5.0,      # L2 regularization
    'objective': 'binary:logistic',  # For ATS prediction
    'eval_metric': 'logloss',   # Or 'error' for accuracy
}

# LightGBM parameters
lgbm_params = {
    'max_depth': 3-5,
    'n_estimators': 300-800,
    'learning_rate': 0.01-0.05,
    'num_leaves': 15-31,        # 2^max_depth - 1 or less
    'subsample': 0.7-0.8,
    'colsample_bytree': 0.7-0.8,
    'min_child_samples': 20-50, # Critical for small datasets
    'reg_alpha': 0.1-1.0,
    'reg_lambda': 1.0-5.0,
    'objective': 'binary',
    'metric': 'binary_logloss',
}
```

**Key insight:** Aggressive regularization is more important than architecture choice.
Use `max_depth=3` until you have strong evidence that deeper trees help on held-out data.
With ~5,000 D1 games/season, you have limited training data relative to many ML applications.

### 1.3 Team Embeddings for 363 NCAA Teams

Three practical approaches, in order of complexity:

**Approach 1: Entity Embeddings via Neural Network (Recommended Starting Point)**

Train a small neural network with an embedding layer for team IDs. The embedding maps each
team's integer ID to a dense vector (typically 8-32 dimensions). Train on game outcomes with
the embedding as part of the feature set.

```python
import torch
import torch.nn as nn

class TeamModel(nn.Module):
    def __init__(self, n_teams=363, embed_dim=16, n_features=20):
        super().__init__()
        self.team_embed = nn.Embedding(n_teams, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2 + n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, home_id, away_id, features):
        home_emb = self.team_embed(home_id)
        away_emb = self.team_embed(away_id)
        x = torch.cat([home_emb, away_emb, features], dim=1)
        return self.fc(x)
```

After training, extract the embeddings and use them as features in your GBM model.
This is a "hybrid" approach that captures team similarity without requiring end-to-end
neural network prediction.

**Approach 2: Node2Vec Graph Embeddings**

Build a game graph where teams are nodes and edges represent games played (weighted by
recency and margin). Run Node2Vec to produce team embeddings. Recent research (2024-2025)
applied this to NCAA tournament data and found that cosine similarity of Node2Vec embeddings
was a statistically significant predictor of March Madness matchup outcomes across 2022-2024
seasons.

```python
from node2vec import Node2Vec
import networkx as nx

# Build game network
G = nx.Graph()
for _, game in games_df.iterrows():
    weight = margin_to_weight(game['margin'])
    G.add_edge(game['home_team'], game['away_team'], weight=weight)

# Generate embeddings
node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200, p=1, q=2)
model = node2vec.fit(window=10, min_count=1)
embeddings = {team: model.wv[team] for team in G.nodes()}
```

**Approach 3: Graph Neural Networks (GNNs)**

The most sophisticated approach. Build a heterogeneous graph with team nodes and game edges.
Use Graph Attention Networks (GAT) or Graph Convolutional Networks (GCN) to learn
context-aware team representations. A July 2025 paper (HIGFormer) introduced a Player-Team
Heterogeneous Interaction Graph Transformer that achieved 52.19% accuracy on NBA outcomes --
modest but notable that it consistently beat all baselines.

**Practical note on team embeddings:** For college basketball specifically, embeddings have
a cold start problem -- with 30% roster turnover via the transfer portal annually, last
season's embedding may not reflect this season's team. Solution: blend preseason embedding
(from returning players) with rapidly updated in-season embedding using the formula
`w = games / (games + k)` where k is approximately 10-15.

### 1.4 Bayesian Hierarchical Models

**Why Bayesian for college basketball:** 363 teams across 32 conferences with vastly
different talent levels. A hierarchical model naturally handles this by partially pooling
team ratings within conferences, borrowing strength from similar teams when data is sparse.

**The PyMC Implementation Pattern:**

Based on the NCAA rating model by Seth Holladay and the PyMC rugby analytics example:

```python
import pymc as pm
import numpy as np

with pm.Model() as ncaa_model:
    # Hyperpriors: conference-level distributions
    # Each conference has its own mean offensive/defensive quality
    conf_off_mu = pm.Normal('conf_off_mu', mu=0, sigma=5, shape=n_conferences)
    conf_def_mu = pm.Normal('conf_def_mu', mu=0, sigma=5, shape=n_conferences)
    conf_off_sigma = pm.HalfNormal('conf_off_sigma', sigma=3)
    conf_def_sigma = pm.HalfNormal('conf_def_sigma', sigma=3)

    # Team-level ratings: drawn from conference distribution
    # This is partial pooling -- small-conference teams shrink toward conf mean
    off_rating = pm.Normal('off_rating',
                           mu=conf_off_mu[team_conference_idx],
                           sigma=conf_off_sigma,
                           shape=n_teams)
    def_rating = pm.Normal('def_rating',
                           mu=conf_def_mu[team_conference_idx],
                           sigma=conf_def_sigma,
                           shape=n_teams)

    # Home court advantage
    home_adv = pm.Normal('home_adv', mu=3.5, sigma=2)

    # Expected score differential
    # Additive model (consistent with KenPom's 2024 update):
    # If Team A offense is +5 and Team B defense is -3,
    # expected margin component = 5 - (-3) = 8
    mu_diff = (off_rating[home_team_idx] - def_rating[away_team_idx]) - \
              (off_rating[away_team_idx] - def_rating[home_team_idx]) + \
              home_adv

    # Score noise
    sigma_score = pm.HalfNormal('sigma_score', sigma=10)

    # Likelihood: observed point differential
    score_diff = pm.Normal('score_diff',
                           mu=mu_diff,
                           sigma=sigma_score,
                           observed=observed_margins)

    # Inference
    trace = pm.sample(3000, tune=1000, target_accept=0.9,
                      return_inferencedata=True)
```

**Key implementation details:**

- **Sum-to-zero constraint:** Add `pm.Deterministic('off_centered', off_rating - off_rating.mean())` to ensure identifiability. Without this, the model can shift all ratings up/down without changing predictions.

- **Conference indexing:** Create an integer mapping from each team to its conference. This is the "hierarchy" -- teams are nested within conferences.

- **Inference method:** NUTS (No-U-Turn Sampler) is the standard for PyMC. Use `target_accept=0.9` or higher for hierarchical models to avoid divergences. Expect 5-15 minutes for a full season on a modern CPU.

- **Prediction with uncertainty:** The posterior gives you distributions, not point estimates. For each game, sample from the posterior to get a distribution of predicted margins. This directly feeds Kelly criterion sizing.

```python
# Generate predictions with uncertainty
with ncaa_model:
    posterior_pred = pm.sample_posterior_predictive(trace)

# Each prediction is a distribution -- use for Kelly sizing
pred_margins = posterior_pred.posterior_predictive['score_diff']
prob_cover = (pred_margins > spread).mean()  # P(team covers)
uncertainty = pred_margins.std()              # Higher = less confident = smaller bet
```

- **Season evolution:** Add a time-varying component using a Gaussian Process or autoregressive term to allow team strength to change during the season. A 2025 approach by Evan Miyakawa's BPR system uses Bayesian state-space models to capture roster changes via the transfer portal.

### 1.5 Ensemble Methods

**The 2025-2026 state of the art for sports prediction ensembles:**

A June 2025 paper in Nature Scientific Reports tested stacked ensembles for NBA prediction.
The stacking architecture used six base learners (XGBoost, KNN, AdaBoost, Naive Bayes,
Logistic Regression, Decision Tree) with an MLP meta-learner. Results:

| Model                | Accuracy | AUC    |
|---------------------|----------|--------|
| XGBoost (alone)     | 81.03%   | 90.82% |
| Logistic Regression | 80.49%   | 90.28% |
| **Stacked Ensemble**| **83.27%** | **92.13%** |

The ensemble improved accuracy by 2+ percentage points over the best individual model,
with statistical significance (p < 0.05).

**Recommended ensemble architecture for this project:**

```
Layer 1 (Base Models):
  - Logistic Regression with Four Factors differentials
  - XGBoost with full feature set
  - LightGBM with full feature set
  - Bayesian hierarchical model (provides uncertainty)
  - Elo rating system (simple, robust baseline)

Layer 2 (Meta-Learner):
  - Logistic Regression or small MLP
  - Trained on out-of-fold predictions from Layer 1
  - Uses 5-fold TEMPORAL cross-validation (not random)
```

**Implementation with scikit-learn:**

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

estimators = [
    ('lr', LogisticRegression(C=1.0, max_iter=1000)),
    ('xgb', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.03)),
    ('lgbm', LGBMClassifier(max_depth=4, n_estimators=500, learning_rate=0.02)),
]

ensemble = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=TimeSeriesSplit(n_splits=5),  # Temporal splits, not random
    stack_method='predict_proba',     # Use probabilities, not labels
    passthrough=False                 # Only use base model predictions
)
```

**Critical note:** Use `TimeSeriesSplit` for the cross-validation within stacking. Standard
k-fold will leak future information into the meta-learner training.

**AutoGluon as an alternative:** Amazon's AutoGluon (v1.5, 2025) now includes a
'zeroshot_2025_tabfm' preset with 22 models including tabular foundation models. It
automatically handles stacking, blending, and hyperparameter tuning. For rapid prototyping:

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label='covers_spread',
    eval_metric='log_loss',
    problem_type='binary'
).fit(train_data, presets='best_quality', time_limit=3600)
```

AutoGluon is useful for establishing an upper bound on what's achievable with your features
before investing time in manual model tuning.

---

## 2. Feature Engineering Deep Dive

### 2.1 Optimal Rolling Window Sizes

Research and practice converge on using **multiple rolling windows as separate features**
rather than picking a single "best" window:

| Window Size | What It Captures | Best For |
|-------------|-----------------|----------|
| 3-5 games   | Very recent form, hot/cold streaks | Detecting short-term momentum shifts |
| 7-10 games  | Medium-term trend | Capturing style changes, lineup adjustments |
| 15-20 games | Stable team identity | Core team quality assessment |
| Full season | Cumulative performance | Baseline rating (regressed toward prior) |

**Implementation pattern:**

```python
def add_rolling_features(df, metric_cols, windows=[5, 10, 15, 20]):
    """Add rolling averages for each metric at each window size."""
    for col in metric_cols:
        for w in windows:
            df[f'{col}_roll_{w}'] = (
                df.groupby('team_id')[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
            )
    return df
```

**Critical:** Always use `.shift(1)` to avoid lookahead bias. The rolling window for game N
must only include data from games 1 through N-1.

**When to use which window size:**

- **Early season (games 1-8):** Only 3-5 game windows have enough data. Blend heavily with
  preseason priors.
- **Mid-season (games 9-20):** All windows become viable. The 10-game window often dominates
  in feature importance.
- **Late season (games 20+):** The 15-20 game window becomes most stable. Short windows still
  capture form but with diminishing marginal value.

### 2.2 Exponential Decay Weighting

EWMA (Exponentially Weighted Moving Average) is generally superior to hard rolling windows
because it provides a smooth weighting function with no arbitrary cutoff.

**Recommended alpha values by metric type:**

| Metric Category | Alpha Range | Half-life (games) | Rationale |
|----------------|-------------|-------------------|-----------|
| Offensive efficiency (ORtg) | 0.92-0.95 | 8-14 games | Stabilizes in ~15-20 games |
| Defensive efficiency (DRtg) | 0.94-0.97 | 12-23 games | Slower to stabilize (~25-30 games) |
| Shooting (eFG%, 3P%) | 0.90-0.93 | 7-10 games | Signal emerges quickly but is noisy |
| Turnover rate (TOV%) | 0.93-0.95 | 10-14 games | Moderate stabilization |
| Rebounding (ORB%) | 0.91-0.94 | 8-12 games | Relatively stable metric |
| Free throw rate (FTr) | 0.90-0.93 | 7-10 games | Style-dependent, stabilizes fast |
| Pace | 0.88-0.92 | 6-9 games | Very stable, coach-determined |

**Implementation:**

```python
def ewma_features(df, metric_cols, alphas=[0.90, 0.93, 0.96]):
    """
    Apply EWMA with multiple decay rates.
    alpha = decay factor (higher = slower decay = more history weight)
    span = 2/(1-alpha) - 1 maps alpha to pandas span parameter
    """
    for col in metric_cols:
        for alpha in alphas:
            span = 2.0 / (1.0 - alpha) - 1.0
            df[f'{col}_ewma_{alpha}'] = (
                df.groupby('team_id')[col]
                .transform(lambda x: x.shift(1).ewm(span=span, min_periods=3).mean())
            )
    return df
```

**Key insight:** Use multiple alpha values as separate features and let the model learn which
decay rate matters for each metric. Do not try to find the "optimal" single alpha -- it varies
by metric and by team.

### 2.3 Opponent Adjustment Methods

**Method 1: Iterative Adjustment (KenPom-style)**

The classic approach. Start with raw efficiency, adjust for opponent quality, re-adjust, repeat.

```python
def iterative_adjustment(games_df, n_iterations=10):
    """Simple iterative opponent adjustment."""
    teams = games_df['team_id'].unique()
    ratings = {t: 0.0 for t in teams}  # Start at average

    for iteration in range(n_iterations):
        new_ratings = {}
        for team in teams:
            team_games = games_df[games_df['team_id'] == team]
            adj_scores = []
            for _, game in team_games.iterrows():
                opp = game['opponent_id']
                # Adjust raw score by opponent quality
                adj_score = game['points_per_poss'] - ratings.get(opp, 0)
                adj_scores.append(adj_score)
            new_ratings[team] = np.mean(adj_scores)
        ratings = new_ratings

    return ratings
```

**Method 2: Ridge Regression (Recommended)**

Solves for all team ratings simultaneously instead of iterating. More statistically principled
and handles small samples better through L2 regularization.

The setup: each game becomes a row. The target is points per possession (or margin). The
features are dummy variables: +1 for the offensive team, -1 for the defensive team, +1 for
home court.

```python
from sklearn.linear_model import RidgeCV
import numpy as np

def ridge_opponent_adjustment(games_df, n_teams):
    """
    Simultaneous opponent adjustment via ridge regression.

    For each game, the model is:
    efficiency = team_off_quality - opp_def_quality + home_advantage + intercept

    This sets up a design matrix with team dummy variables.
    Ridge regularization (L2) shrinks extreme ratings toward the mean,
    which is especially valuable early in the season with small samples.
    """
    n_games = len(games_df)

    # Design matrix: [off_team_dummies | def_team_dummies | home_indicator]
    X = np.zeros((n_games, 2 * n_teams + 1))

    for i, (_, game) in enumerate(games_df.iterrows()):
        X[i, game['team_idx']] = 1           # Offensive team
        X[i, n_teams + game['opp_idx']] = -1 # Defensive team (negative)
        X[i, -1] = game['is_home']           # Home court

    y = games_df['points_per_poss'].values

    # RidgeCV finds optimal alpha via cross-validation
    # Typical optimal alpha: 100-300 for full season, higher for early season
    model = RidgeCV(alphas=[50, 100, 175, 250, 500])
    model.fit(X, y)

    off_ratings = model.coef_[:n_teams]
    def_ratings = model.coef_[n_teams:2*n_teams]
    home_adv = model.coef_[-1]

    return off_ratings, def_ratings, home_adv
```

**Why ridge is better than iteration for this project:**

- Handles early-season small samples gracefully (shrinkage)
- Solves in one pass instead of 10+ iterations
- Natural uncertainty quantification (coefficient standard errors)
- Easily extended: add conference dummy variables, tempo adjustments, etc.
- The regularization parameter alpha naturally increases early in the season when you have
  fewer games, providing more shrinkage when you need it most

**Method 3: Bayesian Opponent Adjustment**

The gold standard. Uses the PyMC hierarchical model described in Section 1.4. Every team
rating comes with a posterior distribution (uncertainty estimate). Early-season ratings are
heavily influenced by the prior (preseason projection), and as games accumulate the posterior
tightens around observed performance.

**KenPom's 2024 Methodology Update:** KenPom switched from a multiplicative to an additive
model for opponent adjustment. Under the new system, if Team A's offense is 10% above average
and Team B's defense is 10% above average (i.e., bad defense), then Team A's expected offense
vs Team B = 20% above average. He also reduced the weight of recency and made game importance
less sensitive to margin and opponent quality.

### 2.4 Handling Small Sample Sizes Early in Season

This is the cold start problem, and it is the single most important practical challenge for
a college basketball model.

**The EvanMiya / Bayesian Performance Rating Approach (2025):**

Each player begins the season with a preseason prior distribution based on:
- Previous college season performance
- High school recruiting rating (for freshmen)
- Transfer portal context (new team, new role)
- Career trajectory (improvement trends)

The prior weight decays as in-season data accumulates. By season's end, preseason priors
carry approximately 15% influence on the final rating. Box score statistics carry over 50%
of the weight for most players through the season.

**Practical implementation for team-level cold start:**

```python
def blend_prior_with_observed(preseason_rating, observed_metric, games_played, k=12):
    """
    Blend preseason prior with observed performance.

    k controls how quickly in-season data dominates:
    - k=10: prior drops to 50% weight after 10 games (~early December)
    - k=12: prior drops to 50% weight after 12 games (~mid December)
    - k=15: prior drops to 50% weight after 15 games (~late December)

    For defensive metrics, use higher k (15-20) because they stabilize slower.
    """
    w = games_played / (games_played + k)
    return w * observed_metric + (1 - w) * preseason_rating


# Different k values for different metric categories
k_values = {
    'off_efficiency': 12,    # Offense stabilizes faster
    'def_efficiency': 18,    # Defense needs more games
    'pace': 8,               # Very stable, converges fast
    'efg_pct': 15,           # Shooting is noisy
    'tov_pct': 12,
    'orb_pct': 12,
    'ftr': 10,
}
```

**Preseason priors -- where to get them:**

- **KenPom preseason ratings** ($20/year): Available before the season starts. Based on
  returning production and recruiting. Well-calibrated because they have been iteratively
  refined over 20+ years.
- **Barttorvik preseason projections** (free): Similar methodology, slightly different weights.
  Available via cbbdata API.
- **EvanMiya team projections** (evanmiya.com): Player-level projections aggregated to team
  level. Uses Bayesian Performance Rating with recruiting data and transfer portal modeling.
- **Vegas preseason futures**: Season win totals imply team-level ratings. Wisdom-of-crowds
  baseline.
- **Build your own**: Use returning production % (% of minutes, points, assists from last
  season that return) combined with recruiting rankings.

### 2.5 Tempo-Free vs Raw Metrics

**Rule: Always use tempo-free (per-possession) metrics for modeling. Use raw metrics only for
specific narrow purposes.**

The classic illustration: VMI scored 78.2 points per game (6th in NCAA) but only 99.3 points
per 100 possessions (189th). Wisconsin scored 65.0 ppg (240th) but 107.6 per 100 possessions
(64th). VMI's up-tempo pace inflated their raw numbers; Wisconsin was far more efficient.

**When raw stats are acceptable:**

- Total points scored (for totals markets -- you are literally predicting raw points)
- Minutes played (for player prop markets)
- Pace itself is a raw metric (possessions per 40 minutes) and is important as a feature

**When you must use tempo-free:**

- Any model predicting spread or win probability
- Any comparative analysis across teams
- Feature engineering for efficiency metrics
- Any opponent adjustment calculation

### 2.6 Player Availability Impact Quantification

**Minutes-weighted approach:**

```python
def estimate_injury_impact(team_df, injured_player, replacement_player):
    """
    Estimate point differential impact of a player substitution.

    Uses minutes-weighted contribution approach:
    impact = (injured_BPR - replacement_BPR) * (minutes_share / 100)

    BPR = Bayesian Performance Rating (points per 100 possessions above average)
    minutes_share = % of team possessions the injured player was on court
    """
    injured_bpr = team_df.loc[injured_player, 'bpr']
    replacement_bpr = team_df.loc[replacement_player, 'bpr']
    minutes_share = team_df.loc[injured_player, 'poss_pct']

    # Expected margin impact per game (points)
    # Divide by 100 because BPR is per 100 possessions
    # Multiply by team pace to convert to per-game
    poss_per_game = team_df['pace'].mean()
    impact = (injured_bpr - replacement_bpr) * (minutes_share / 100) * (poss_per_game / 100)

    return impact
```

**Key findings from recent research (2024-2025):**

- Most injury types produce measurable performance declines in 2-, 5-, and 10-game windows
  post-return, meaning even "healthy" returning players underperform initially.
- The impact is strongest in offensive and defensive ratings.
- Teams with balanced rest management perform better overall.
- For college basketball specifically, the impact of losing a top player is amplified because
  roster depth is typically much shallower than in the NBA.

### 2.7 Travel and Rest Fatigue Modeling

**What the research says (from studies of 8,500+ NBA games and 25,000+ match analysis):**

| Factor | Impact | Confidence |
|--------|--------|------------|
| Back-to-back games | -1.5 to -2.5 points for traveling team | High |
| 1 day rest vs 2 days | +1.1 home pts, +1.6 away pts with 2 days | High |
| 3+ days rest (optimal) | Peak performance, then declines after 4+ days | Medium |
| Eastward travel | -2 to -3 points vs staying home | High |
| Westward travel | -1 to -2 points vs staying home | Medium |
| Cross-timezone (PDT home vs EDT away) | PDT team wins 63.5% at home | High |
| Away-to-Home sequence | 54.4% win rate (best travel sequence) | Medium |

**College basketball-specific adjustments:**

- College teams travel less frequently than NBA teams, so travel effects are amplified when
  they do occur (less acclimatization).
- Mid-major conference road trips often involve back-to-back games on Thursday-Saturday.
- November/December non-conference scheduling involves unusual travel patterns (exempt
  tournaments in resort locations, true road games at major programs).
- Exam periods (early December, early May) may affect player focus and practice time.

**Feature engineering for travel/rest:**

```python
def travel_features(schedule_df):
    """Compute travel and rest features for each game."""
    features = {}

    features['days_rest'] = (game_date - prev_game_date).days
    features['is_back_to_back'] = features['days_rest'] <= 1
    features['travel_distance_miles'] = haversine(prev_venue, current_venue)
    features['timezone_change'] = abs(current_tz - prev_tz)
    features['travel_direction'] = 'east' if current_tz > prev_tz else 'west'
    features['games_in_last_7_days'] = count_games_in_window(7)
    features['games_in_last_14_days'] = count_games_in_window(14)
    features['is_road_trip'] = consecutive_away_games > 1
    features['road_trip_game_number'] = nth_away_game_in_streak

    return features
```

---

## 3. Validation and Evaluation

### 3.1 Walk-Forward Cross-Validation

**This is non-negotiable.** Standard k-fold cross-validation is invalid for temporal sports
data because it leaks future information into the training set.

**Implementation:**

```python
def walk_forward_cv(data, model_class, min_train_seasons=3):
    """
    Walk-forward validation for sports prediction.

    Train on seasons 1-N, predict season N+1.
    Slide forward one season at a time.
    """
    seasons = sorted(data['season'].unique())
    results = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train = data[data['season'].isin(train_seasons)]
        test = data[data['season'] == test_season]

        model = model_class()
        model.fit(train[features], train[target])

        preds = model.predict_proba(test[features])[:, 1]

        results.append({
            'season': test_season,
            'accuracy': accuracy_score(test[target], preds > 0.5),
            'log_loss': log_loss(test[target], preds),
            'brier_score': brier_score_loss(test[target], preds),
            'auc': roc_auc_score(test[target], preds),
            'n_games': len(test),
        })

    return pd.DataFrame(results)
```

**Intra-season validation (more granular):**

For within-season model updates, use an expanding window: train on all games played so far
in the season, predict the next week's games.

```python
def intra_season_validation(season_data, model, retrain_weekly=True):
    """
    Expanding window within a single season.
    Mimics real-world usage where you retrain as new games come in.
    """
    weeks = sorted(season_data['week'].unique())
    all_preds = []

    for week in weeks[4:]:  # Start predicting after 4 weeks
        train = season_data[season_data['week'] < week]
        test = season_data[season_data['week'] == week]

        if retrain_weekly:
            model.fit(train[features], train[target])

        preds = model.predict_proba(test[features])[:, 1]
        all_preds.extend(zip(test.index, preds, test[target]))

    return all_preds
```

### 3.2 Calibration

**Calibration is more important than discrimination for betting.** A well-calibrated model
assigns probabilities that match observed frequencies. If you predict 65% probability, the
event should happen 65% of the time.

A 2024 study found that optimizing for calibration instead of accuracy led to **69.86% higher
average returns** in simulated betting.

**Measuring calibration:**

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def evaluate_calibration(y_true, y_prob, n_bins=10):
    """
    Compute calibration metrics and plot reliability diagram.

    ECE (Expected Calibration Error): weighted average |predicted - actual| per bin
    MCE (Maximum Calibration Error): worst bin's |predicted - actual|
    Lower is better for both. ECE < 0.015 is excellent.
    """
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )

    # Expected Calibration Error
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    weights = bin_counts / bin_counts.sum()
    ece = np.sum(weights * np.abs(fraction_positive - mean_predicted))

    # Maximum Calibration Error
    mce = np.max(np.abs(fraction_positive - mean_predicted))

    return ece, mce
```

**Calibration methods (post-hoc):**

| Method | When to Use | Pros | Cons |
|--------|------------|------|------|
| Platt Scaling | Default choice, especially for GBMs | Fast, works with nightly updates | May underfit complex distributions |
| Isotonic Regression | When you have 1000+ validation samples | Non-parametric, captures any pattern | Overfits with small datasets |
| Temperature Scaling | When model is roughly calibrated already | Single parameter, hard to overfit | Limited flexibility |
| Beta Calibration | When predictions cluster at extremes | Handles tail skew | More parameters to fit |

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling (logistic calibration)
calibrated_model = CalibratedClassifierCV(
    base_model, method='sigmoid', cv=5  # 'sigmoid' = Platt scaling
)
calibrated_model.fit(val_features, val_target)

# Isotonic regression calibration
calibrated_model_iso = CalibratedClassifierCV(
    base_model, method='isotonic', cv=5
)
```

**Recommended workflow:**

1. Train raw model on training data
2. Generate predictions on held-out calibration set (different from test set)
3. Fit Platt scaling on calibration predictions
4. Apply calibration transform to test predictions
5. Verify calibration with reliability diagram
6. Monitor ECE monthly during live betting -- if ECE > 0.015, re-fit calibration

### 3.3 Metrics That Matter for Betting

**Log Loss vs Brier Score vs AUC:**

| Metric | What It Measures | Betting Relevance |
|--------|-----------------|-------------------|
| **Log Loss** | Penalizes confident wrong predictions heavily | Best for betting -- directly penalizes overconfidence, which destroys bankroll |
| **Brier Score** | Mean squared error of probabilities | Second best -- measures calibration but less sensitive to extreme miscalibrations |
| **AUC** | Discrimination (ranking) ability | Least useful for betting -- you can have perfect AUC and terrible calibration |
| **Accuracy** | Binary correct/wrong | Nearly useless -- ignores probability quality entirely |

**The August 2025 NCAA paper (Habib) confirmed this distinction:**
- Transformer + BCE loss: Highest AUC (0.8473) but worse calibration
- LSTM + Brier loss: Best calibration (Brier 0.1589) but lower AUC
- For betting, the LSTM was the better model despite lower AUC

**Bottom line:** Optimize for log loss or Brier score, not AUC or accuracy. Report AUC as a
secondary metric for discrimination assessment.

### 3.4 Measuring Improvement at Each Feature Step

Follow the incremental approach from CLAUDE.md. At each step:

```python
def evaluate_feature_addition(base_features, new_feature, data, model_class):
    """
    Measure marginal improvement from adding a feature.
    Uses paired comparison on the same test sets.
    """
    results_base = walk_forward_cv(data, model_class, features=base_features)
    results_new = walk_forward_cv(data, model_class, features=base_features + [new_feature])

    improvement = {
        'log_loss_delta': results_base['log_loss'].mean() - results_new['log_loss'].mean(),
        'brier_delta': results_base['brier_score'].mean() - results_new['brier_score'].mean(),
        'is_significant': paired_t_test(results_base['log_loss'], results_new['log_loss']),
    }

    return improvement
```

**Decision rule:** Add the feature if:
1. Log loss improves by > 0.005 on held-out data
2. Improvement is consistent across seasons (not just one outlier year)
3. The feature has a plausible causal mechanism
4. Adding it does not substantially increase overfitting (train-test gap)

### 3.5 CLV (Closing Line Value) as Evaluation Metric

**CLV is the single most important live performance metric.** It measures whether your bets
consistently obtain better odds than the closing line, which represents the market's most
accurate probability estimate.

```python
def calculate_clv(bets_df):
    """
    Calculate Closing Line Value for a set of bets.

    CLV = (closing_implied_prob - opening_implied_prob) / opening_implied_prob

    Positive CLV means you consistently bet at better prices than the close.
    If you can't beat the closing line, your model has no edge.
    """
    bets_df['bet_implied_prob'] = american_to_implied(bets_df['bet_odds'])
    bets_df['close_implied_prob'] = american_to_implied(bets_df['closing_odds'])

    # Positive means you got better odds than close
    bets_df['clv'] = (bets_df['close_implied_prob'] - bets_df['bet_implied_prob']) / \
                      bets_df['bet_implied_prob']

    avg_clv = bets_df['clv'].mean()
    clv_positive_pct = (bets_df['clv'] > 0).mean()

    return avg_clv, clv_positive_pct


def american_to_implied(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)
```

**CLV benchmarks:**
- Average CLV > 0%: You have some edge
- Average CLV > 1%: Solid edge, likely profitable long-term
- Average CLV > 3%: Elite performance
- Average CLV < 0%: Your model is not beating the market -- do not bet

---

## 4. Practical Implementation

### 4.1 Python Library Stack

**Core libraries:**

```
# requirements.txt for this project

# Data collection
cbbpy>=2.0              # NCAA box scores and PBP from ESPN
sportsdataverse         # Broader sports data access
requests                # For API calls (Odds API, cbbdata)

# Data processing
pandas>=2.0
numpy>=1.24
polars>=0.20            # Faster than pandas for large datasets

# Feature engineering
scikit-learn>=1.3
scipy

# Modeling
xgboost>=2.0
lightgbm>=4.0
pymc>=5.10              # Bayesian models (uses PyTensor backend)
# Stan via cmdstanpy    # Alternative Bayesian backend (faster sampling)

# Calibration and evaluation
scikit-learn            # CalibratedClassifierCV, calibration_curve

# Embeddings (optional, for advanced architectures)
torch>=2.0              # Team embeddings, neural components
node2vec                # Graph embeddings

# AutoML (optional, for benchmarking)
autogluon.tabular       # Automated ensemble learning

# Visualization
matplotlib
seaborn
plotly                  # Interactive reliability diagrams

# Scheduling and pipeline
schedule                # For daily cron-like updates
sqlalchemy              # Database interface for feature store
```

**On PyMC vs Stan:**

PyMC 5 (current stable) uses PyTensor (formerly Aesara) as its computation backend and the
NUTS sampler. It is pure Python and integrates cleanly with the rest of the stack.

Stan (via cmdstanpy) is faster for sampling, especially with large hierarchical models.
If your Bayesian model takes >30 minutes to sample in PyMC, consider porting to Stan.
However, the development experience in PyMC is significantly better.

### 4.2 Data Pipeline Architecture

```
Daily Update Pipeline (during season):

1. COLLECT (6:00 AM ET)
   ├── Pull yesterday's box scores via cbbpy
   ├── Pull yesterday's PBP data via cbbpy
   ├── Pull updated odds via The Odds API
   ├── Pull updated team ratings via cbbdata API (Barttorvik)
   └── Check injury reports (manual or ESPN scrape)

2. PROCESS (6:30 AM ET)
   ├── Compute game-level tempo-free statistics
   ├── Update rolling windows and EWMA features
   ├── Run opponent adjustment (ridge regression)
   ├── Update team embeddings (incremental)
   └── Store all features in feature store (SQLite/PostgreSQL)

3. PREDICT (7:00 AM ET)
   ├── Load today's schedule and current lines
   ├── Generate features for each game
   ├── Run ensemble model predictions
   ├── Apply calibration transform
   ├── Calculate edge: model_prob vs implied_prob
   └── Size bets using fractional Kelly

4. EXECUTE (varies by market)
   ├── Check exchange liquidity (STX)
   ├── Place orders at target prices
   └── Log all bets with timestamps and odds

5. EVALUATE (next morning)
   ├── Record results for yesterday's bets
   ├── Calculate CLV for each bet
   ├── Update running performance metrics
   └── Flag model drift warnings
```

**Scheduling in Python:**

```python
import schedule
import time

def daily_pipeline():
    collect_data()
    process_features()
    generate_predictions()
    execute_bets()

schedule.every().day.at("06:00").do(daily_pipeline)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 4.3 Feature Store Design

Use a simple SQL-based feature store. For a solo project, SQLite is sufficient.
PostgreSQL if you want concurrent access or plan to scale.

```python
import sqlite3
import pandas as pd

class FeatureStore:
    def __init__(self, db_path='features.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS team_features (
                team_id TEXT,
                as_of_date DATE,
                feature_name TEXT,
                feature_value REAL,
                PRIMARY KEY (team_id, as_of_date, feature_name)
            )
        ''')

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS game_features (
                game_id TEXT,
                feature_name TEXT,
                feature_value REAL,
                PRIMARY KEY (game_id, feature_name)
            )
        ''')

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                game_id TEXT,
                model_name TEXT,
                prediction_date DATETIME,
                predicted_prob REAL,
                calibrated_prob REAL,
                market_implied_prob REAL,
                edge REAL,
                kelly_fraction REAL,
                PRIMARY KEY (game_id, model_name, prediction_date)
            )
        ''')

    def get_team_features(self, team_id, as_of_date, feature_names=None):
        """Retrieve features for a team as of a specific date.
        This ensures no lookahead bias -- you only get data available at that time."""
        query = '''
            SELECT feature_name, feature_value
            FROM team_features
            WHERE team_id = ? AND as_of_date <= ?
            AND as_of_date = (
                SELECT MAX(as_of_date) FROM team_features
                WHERE team_id = ? AND as_of_date <= ?
                AND feature_name = team_features.feature_name
            )
        '''
        df = pd.read_sql(query, self.conn, params=[team_id, as_of_date,
                                                     team_id, as_of_date])
        return df.set_index('feature_name')['feature_value'].to_dict()
```

**Critical design principle:** The feature store must enforce temporal ordering. When you
query features for a game on January 15, you must only get data from January 14 or earlier.
The `as_of_date <= ?` pattern above enforces this.

### 4.4 Cold Start Strategy

**Season timeline and model behavior:**

| Period | Games Played | Model Strategy |
|--------|-------------|----------------|
| Preseason (Oct) | 0 | Use only preseason priors (KenPom, Barttorvik, EvanMiya) |
| Early Nov | 1-3 | 80% prior, 20% observed. Use high regularization alpha |
| Late Nov | 4-8 | 60% prior, 40% observed. Start using rolling windows |
| December | 9-12 | 40% prior, 60% observed. All features become viable |
| January | 13-18 | 20% prior, 80% observed. Model approaching full confidence |
| Feb-March | 19+ | <15% prior. Priors mainly help for opponent adjustment |

**Practical implementation:**

```python
def get_season_phase_config(games_played):
    """Return model configuration based on games played."""
    if games_played <= 3:
        return {
            'prior_weight': 0.80,
            'rolling_windows': [3],
            'regularization_mult': 3.0,  # 3x normal regularization
            'min_confidence_for_betting': 0.08,  # Need larger edge to bet
        }
    elif games_played <= 8:
        return {
            'prior_weight': 0.60,
            'rolling_windows': [3, 5],
            'regularization_mult': 2.0,
            'min_confidence_for_betting': 0.06,
        }
    elif games_played <= 15:
        return {
            'prior_weight': 0.35,
            'rolling_windows': [5, 10],
            'regularization_mult': 1.5,
            'min_confidence_for_betting': 0.04,
        }
    else:
        return {
            'prior_weight': 0.15,
            'rolling_windows': [5, 10, 15, 20],
            'regularization_mult': 1.0,
            'min_confidence_for_betting': 0.03,
        }
```

### 4.5 Model Retraining Frequency

**Recommended approach: hybrid periodic + drift-triggered retraining.**

| Action | Frequency | What Changes |
|--------|-----------|-------------|
| Feature update | Daily | Rolling windows, EWMA, opponent adjustments |
| Calibration check | Weekly | Re-fit Platt scaling if ECE drifts > 0.015 |
| Full model retrain | Monthly | Re-fit GBM hyperparameters on all available data |
| Architecture review | Seasonal | Add/remove features, try new model types |
| Prior update | Annually (preseason) | New preseason projections for next season |

**Drift detection:**

```python
def check_model_drift(recent_preds, window=100):
    """
    Check if model calibration has drifted.
    Triggers retraining if ECE exceeds threshold.
    """
    if len(recent_preds) < window:
        return False

    last_n = recent_preds.tail(window)
    ece, _ = evaluate_calibration(last_n['actual'], last_n['predicted'])

    if ece > 0.015:
        logging.warning(f"Model drift detected: ECE = {ece:.4f}")
        return True

    return False
```

---

## 5. Common Mistakes and How to Avoid Them

### 5.1 Overfitting with Too Many Features

**Rule of thumb:** No more than 1 feature per 10-20 training observations.

With ~5,000 D1 games/season and 3 seasons of training data (15,000 games), that gives you
a budget of 75-150 features at the 10:1 ratio, or 750-1,500 at the 20:1 ratio.

However, effective degrees of freedom matter more than raw feature count. Correlated features
(like eFG% and ORtg) do not add independent information. Use feature importance and ablation
studies to prune.

**Practical checks:**
- If train accuracy >> test accuracy by more than 3-5%, you are overfitting
- If adding a feature improves train performance but not test performance, remove it
- Use permutation importance (not built-in GBM importance) to identify truly useful features

### 5.2 Target Leakage in Feature Engineering

**The most insidious form of leakage in sports prediction is temporal leakage.**

Common examples in basketball:
- Using season-average eFG% to predict mid-season games (includes future games)
- Computing opponent adjustment using the full season's data for early-season games
- Fitting scaler/normalizer on the full dataset before train-test split
- Using features that include the game being predicted (forgetting `.shift(1)`)

**Prevention checklist:**

1. Always use `.shift(1)` or filter by `game_date < prediction_date`
2. Fit all transformers (scalers, encoders) only on training data
3. Run opponent adjustments using only games played before the prediction date
4. When in doubt, simulate: "Could I actually compute this feature at 6 AM on game day,
   using only publicly available information?"
5. Test for leakage: if your model achieves >70% accuracy on ATS prediction, something is
   almost certainly leaked

### 5.3 Ignoring the Closing Line

**If your model cannot beat the closing line, it has no edge.** Period.

The closing line is the most efficient predictor of game outcomes because it incorporates
all publicly available information plus the collective wisdom of sharp bettors. A model that
simply predicts "the closing spread is correct" will beat most ML models.

**How to use this insight:**
- Always benchmark your model against "predict the closing line" as a baseline
- Track CLV (Section 3.5) for every bet you place
- If your CLV is negative over 200+ bets, your model is not adding value
- The market is not always right, but you need to prove that you are right more often

### 5.4 Overconfidence in Backtest Results

**Backtests systematically overstate live performance for several reasons:**

1. **Execution slippage:** Backtests assume you get the odds you want. In practice, lines
   move, liquidity is limited, and on exchanges you pay commission (STX: ~2%).

2. **Selection bias:** You test 50 model variations and report the best one. This guarantees
   a fake edge through multiple hypothesis testing.

3. **Survivorship bias in data:** Historical odds data may not include all games, or may
   have survivorship-biased odds from books that went under.

4. **Market movement:** Even on exchanges, placing a large bet can move the line against
   you, especially in low-liquidity college basketball markets.

**Mitigation strategies:**
- Apply a 1-2% haircut to all backtest returns to simulate slippage
- Use Bonferroni correction or false discovery rate control when testing multiple strategies
- Paper trade for at least 200 bets before risking real money
- Start with minimum stakes to verify live execution matches backtest assumptions

### 5.5 Not Accounting for Market Execution

**On exchanges like STX specifically:**

- Commission is typically 2% on net winnings
- Not all games have active markets (especially small-conference NCAAB)
- Liquidity may be thin -- you may not get matched at your desired price
- College basketball markets on STX primarily cover spreads and moneylines for bigger games

**Adjustments for realistic backtesting:**

```python
def apply_exchange_costs(edge, commission_rate=0.02):
    """
    Adjust calculated edge for exchange commission.
    Only bet if edge exceeds commission + minimum threshold.
    """
    net_edge = edge - commission_rate
    min_edge_threshold = 0.02  # 2% minimum net edge to bet

    if net_edge > min_edge_threshold:
        return net_edge
    else:
        return 0  # No bet
```

---

## 6. What's New in 2025-2026

### 6.1 Recent Papers and Research

**Key papers:**

- **Habib (August 2025)** -- "Forecasting NCAA Basketball Outcomes with Deep Learning: A
  Comparative Study of LSTM and Transformer Models" (arXiv:2508.02725). Compared LSTM and
  Transformer architectures for NCAA tournament prediction. Found that LSTM with Brier loss
  training produced better-calibrated probabilities (Brier 0.1589) while Transformers with
  BCE had better discrimination (AUC 0.8473). Key takeaway: loss function choice matters as
  much as architecture choice.

- **Nature Scientific Reports (June 2025)** -- Stacked ensemble for NBA prediction using
  6 base learners + MLP meta-learner. Achieved 83.27% accuracy (AUC 92.13%), outperforming
  all individual models by 2+ percentage points with statistical significance.

- **PLOS One (June 2025)** -- Systematic review of AI techniques for basketball prediction.
  Found that tree-based methods still dominate but neural approaches (LSTM, GNN) are
  increasingly competitive. Noted that Vegas spread prediction error has increased from
  9.12 points (2006-2016) to 10.49 points (2020-2026), coinciding with increased three-point
  shooting variance.

- **HIGFormer (July 2025)** -- Player-Team Heterogeneous Interaction Graph Transformer.
  Novel architecture combining player-level and team-level graph networks with transformer
  attention for game outcome prediction.

- **EvanMiya BPR upgrades (2025)** -- Major updates to the Bayesian Performance Rating
  model, now considered the leading player-level metric in college basketball. The model
  combines Bayesian RAPM, Box Plus-Minus trained on D1 data, and preseason recruiting
  priors into a single framework.

- **Kelly Betting as Bayesian Model Evaluation (February 2026, arXiv)** -- New framework
  connecting Kelly criterion directly to Bayesian model selection, providing a formal
  justification for fractional Kelly based on posterior uncertainty.

### 6.2 New Data Sources and APIs

- **cbbdata API** (2024-2025): Flask-based API providing Barttorvik data with 30+ endpoints,
  updated every 15 minutes during the season. Replaces the older toRvik R package. Free API
  key. Primarily accessed via R package but the backend is Python/REST.

- **BALLDONTLIE NCAAB API**: New API providing NCAA basketball data including betting odds
  endpoints. Supports Python.

- **Unabated API**: WebSocket-based real-time odds feed covering college basketball. Premium
  but provides sub-second odds updates for CLV tracking.

- **The Odds API**: Now covers NCAAB with historical odds data back to late 2020. REST API
  with Python examples. Pricing starts at $20/month for 500 requests.

- **AutoGluon 1.5 (2025)**: Amazon's AutoML framework now includes tabular foundation models
  in its 'zeroshot_2025_tabfm' preset, providing state-of-the-art ensembles with minimal
  configuration.

### 6.3 Changes in Market Efficiency

**The market is getting sharper, but college basketball retains structural inefficiencies.**

Key observations from 2024-2026:

- **Sportsbooks now use real-time AI** to update lines continuously during games. Pre-game
  lines are set with more sophisticated models than even 2-3 years ago.

- **However**, the data above (Vegas spread error increasing from 9.12 to 10.49 points)
  suggests the game itself has become harder to predict due to increased three-point
  variance, not that lines are more accurate.

- **College basketball remains structurally inefficient** because:
  - Analyst coverage still concentrates on 60 Power conference teams
  - Transfer portal creates unprecedented year-to-year roster volatility
  - Small conference games still set largely by algorithm with minimal human oversight
  - Early season lines (November-December) remain significantly softer

- **Exchange markets** (STX, Novig) are still developing liquidity in NCAAB. This creates
  both opportunities (less sharp pricing) and challenges (thin liquidity, wide spreads on
  smaller games).

- **The AI arms race is real:** If you are not using ML/statistical methods, you are at a
  disadvantage. But the floor has risen -- basic models that would have been profitable in
  2015 may not be in 2026.

### 6.4 Impact of AI on Market Sharpness

Top AI models beat closing lines by 3-7% on average across sports, but this overstates the
achievable edge for an individual bettor because:

1. Those models may be market-making the closing line itself
2. Execution friction (slippage, commission, liquidity) eats into the edge
3. Multiple AI-powered bettors competing reduces the edge available to each

**Realistic edge expectation in 2026:** 1-3% ROI on volume for a good individual model,
with the higher end achievable in less efficient sub-markets (small-conference NCAAB, first
half totals, player props where available on exchanges).

---

## 7. Canada-Specific Considerations

### 7.1 Legal Landscape

**Ontario is the primary regulated market for this project.** Since April 2022, Ontario has
operated a regulated iGaming market overseen by the Alcohol and Gaming Commission of Ontario
(AGCO) and iGaming Ontario. As of 2026, 35+ online sportsbooks are licensed.

**Single-event betting** has been legal across Canada since August 2021 (Bill C-218).
Provincial lotteries operate sportsbooks (Proline+ in Ontario, Mise-o-jeu in Quebec, etc.),
and Ontario additionally allows private operators.

**Alberta** is expected to launch its regulated market in early 2026 under the iGaming
Alberta Act.

### 7.2 Exchange Betting in Canada

**STX (Sports Trading Exchange)** is the only betting exchange currently operating in
Canada, licensed by AGCO for Ontario.

Key details:
- First betting exchange to launch in Canada (2022-2023)
- Peer-to-peer model: users back or lay outcomes, setting their own prices
- NCAA basketball markets available (spreads, moneylines for major games)
- Coverage is strongest for high-profile college basketball games; small-conference coverage
  may be limited
- Commission structure similar to other exchanges (~2% on net winnings)
- No public API documentation found as of February 2026 -- manual order placement required

**November 2025 ruling:** The Ontario Court of Appeal ruled 4-1 that Ontarians can participate
in peer-to-peer games and betting with players outside Canada, opening the door for
international exchange access.

**Implications for this project:**
- STX is the primary execution venue for peer-to-peer NCAAB betting in Canada
- Liquidity will be a constraint, especially for smaller games
- Monitor whether Betfair, Sporttrade, or other exchanges enter the Ontario market following
  the November 2025 ruling
- Without an API, automated bet execution is not currently feasible on STX -- manual
  execution introduces latency risk

### 7.3 Tax Considerations

In Canada, gambling winnings are generally **tax-free** for recreational bettors. However,
if betting constitutes a business activity (systematic, profit-seeking, with ML models and
daily pipelines), the CRA may classify it as business income, which is fully taxable.

Consult a Canadian tax professional familiar with gambling income classification. Keep
detailed records regardless.

---

## 8. Sources

### Papers and Academic Research

- [Uncertainty-Aware Machine Learning for NBA Forecasting in Digital Betting Markets (January 2026)](https://www.mdpi.com/2078-2489/17/1/56)
- [Forecasting NCAA Basketball Outcomes with Deep Learning: LSTM vs Transformer (August 2025)](https://arxiv.org/abs/2508.02725)
- [Stacked Ensemble Model for NBA Game Outcome Prediction (June 2025)](https://www.nature.com/articles/s41598-025-13657-1)
- [The Application of AI Techniques in Predicting Basketball Outcomes: A Systematic Review (June 2025)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0326326)
- [A Systematic Review of Machine Learning in Sports Betting (October 2024)](https://arxiv.org/html/2410.21484v1)
- [Machine Learning for Sports Betting: Calibration vs Accuracy (2024)](https://www.sciencedirect.com/science/article/pii/S266682702400015X)
- [Player-Team Heterogeneous Interaction Graph Transformer (July 2025)](https://arxiv.org/pdf/2507.10626)
- [Kelly Betting as Bayesian Model Evaluation (February 2026)](https://arxiv.org/html/2602.09982)
- [Modified Kelly Criteria (Chu, Wu, Swartz)](https://www.sfu.ca/~tswartz/papers/kelly.pdf)
- [Impacts of Travel Distance and Direction on NBA Back-to-Back Games](https://pmc.ncbi.nlm.nih.gov/articles/PMC8636381/)
- [Time Zones and Tiredness Influence NBA Results (May 2024)](https://www.sciencedaily.com/releases/2024/05/240501091642.htm)
- [Sports Analytics for Evaluating Injury Impact on NBA Performance](https://www.mdpi.com/2078-2489/16/8/699)
- [Deep Similarity Learning for Sports Team Ranking](https://arxiv.org/pdf/2103.13736)
- [Graph Neural Networks to Predict Sports Outcomes](https://arxiv.org/abs/2207.14124)

### Methodologies and Technical Resources

- [Bayesian Performance Rating: The Best Player Metric in CBB (EvanMiya)](https://blog.evanmiya.com/p/bayesian-performance-rating)
- [Preseason Player Projections With Bayesian Performance Rating](https://blog.evanmiya.com/p/preseason-player-projections-with)
- [Opponent-Adjusted Stats Using Ridge Regression (CFBD Blog)](https://blog.collegefootballdata.com/opponent-adjusted-stats-ridge-regression/)
- [KenPom Ratings Methodology Update](https://kenpom.com/blog/ratings-methodology-update/)
- [Rating College Basketball Teams with Probabilistic Programming (Seth Holladay)](https://sethah.github.io/ncaa-ratings.html)
- [AI Model Calibration for Sports Betting: Brier Score and Reliability](https://www.sports-ai.dev/blog/ai-model-calibration-brier-score)
- [Log Loss vs. Brier Score (DRatings)](https://www.dratings.com/log-loss-vs-brier-score/)
- [Why Fractional Kelly? Simulations of Bet Size with Uncertainty](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html)
- [Developing Hierarchical Models for Sports Analytics (PyMC Labs)](https://www.pymc-labs.com/blog-posts/2023-09-15-Hierarchical-models-Chris-Fonnesbeck)
- [PyMC Hierarchical Rugby Prediction Example](https://www.pymc.io/projects/examples/en/latest/case_studies/rugby_analytics.html)
- [PyMC NBA Item Response Theory Example](https://github.com/pymc-devs/pymc-examples/blob/main/examples/case_studies/item_response_nba.ipynb)
- [Fitting It In: Adjusting Team Metrics for Schedule Strength (Google Cloud)](https://medium.com/analyzing-ncaa-college-basketball-with-gcp/fitting-it-in-adjusting-team-metrics-for-schedule-strength-4e8239be0530)

### Data Sources and APIs

- [CBBpy: Python NCAA Basketball Scraper](https://pypi.org/project/CBBpy/)
- [cbbdata API: College Basketball Data](https://cbbdata.aweatherman.com/articles/release.html)
- [The Odds API: NCAA Basketball Odds](https://the-odds-api.com/sports-odds-data/ncaa-basketball-odds.html)
- [SportsDataverse Python Package](https://sportsdataverse-py.sportsdataverse.org/)
- [EvanMiya CBB Analytics](https://evanmiya.com/)
- [AutoGluon Tabular Documentation](https://auto.gluon.ai/stable/tutorials/tabular/tabular-indepth.html)

### Betting and Exchange Platforms

- [STX Sports Trading Exchange (Ontario)](https://stxapp.ca/)
- [STX Review and Detailed Overview (next.io)](https://next.io/betting-sites-on/stx/)
- [Ontario Court of Appeal: International P2P Gaming Ruling (November 2025)](https://www.blakes.com/insights/ontario-court-of-appeal-green-lights-access-to-international-pooled-liquidity-for-online-gaming-in-o/)
- [Closing Line Value Explained (Pinnacle)](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)
- [Backtesting a Sports Betting Strategy (Systematic Sports)](https://medium.com/systematic-sports/backtesting-a-sports-betting-strategy-283833a5eca3)

### Market and Industry Analysis

- [How AI Is Rewiring Modern Sports Betting (DeGroote School of Business, McMaster)](https://degroote.mcmaster.ca/articles/how-ai-is-rewiring-modern-sports-betting/)
- [AI in Sports Betting: 2026 Trends (VegasInsider)](https://www.vegasinsider.com/sportsbooks/sports-betting-news/rise-of-ai-in-sports-betting/)
- [Canada Sports Betting Legal Guide 2026](https://www.oddsshark.com/canada)
