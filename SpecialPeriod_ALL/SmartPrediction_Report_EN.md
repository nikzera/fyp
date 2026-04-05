# Smart Prediction: Automated Period Matching & Model Selection

> **Module**: `SpecialPeriod.py` — Smart Predict Mode (Mode 2)  
> **Purpose**: Given an arbitrary date range, automatically identify its market regime, select the optimal model ensemble, and generate predictions.

---

## 1. Motivation

In the Classic Mode of `SpecialPeriod.py`, users must manually select models, markets, and periods. This requires domain expertise to know which models perform well under which market conditions. **Smart Prediction** automates this entire process:

- The user only provides a **start date**, **end date**, and **market**.
- The system determines what type of market regime the period resembles.
- It selects and weights the best-performing models accordingly.
- It trains, predicts, and outputs results — fully automated.

---

## 2. System Architecture

```
User Input: start_date, end_date, market
                    |
    [Step 1] Feature Extraction
        Extract 6-dimensional statistical feature vector
        from actual market data within the date range
                    |
    [Step 2] Period Matching
        Compare against 7 predefined Special Periods
        using z-score normalized Euclidean distance
                    |
    [Step 3] Category Mapping
        Map matched period -> one of 4 regime categories
                    |
    [Step 4] Model Selection (per horizon)
        Use market-specific, horizon-aware rules
        to select models and ensemble weights
                    |
    [Step 5] Train & Predict
        Walk-forward training -> weighted ensemble prediction
                    |
    Output: predictions, curves, SmartPredict_Summary.csv
```

---

## 3. Feature Extraction

For any given date range, the system extracts a **6-dimensional feature vector** from actual market price data:

| # | Feature | Formula | Purpose |
|---|---------|---------|---------|
| 1 | **Mean Return** | `mean(log_returns)` | Distinguish bull / bear / sideways |
| 2 | **Volatility** | `std(log_returns)` | Separate high-volatility crises from calm trends |
| 3 | **Max Drawdown** | `max(running_max(cum_ret) - cum_ret)` | Measure severity of decline |
| 4 | **Skewness** | `skew(log_returns)` | Left-skew = frequent sharp drops; right-skew = sharp rallies |
| 5 | **Kurtosis** | `kurtosis(log_returns)` | High kurtosis = fat tails / black swan events |
| 6 | **Trend Strength** | `sum(log_returns) / volatility` | Sharpe-like ratio; directional conviction |

These features are computed from the **Log Return** column (first feature in the scaled data array), ensuring consistency with the training pipeline.

---

## 4. Period Matching Algorithm

### 4.1 Reference Periods

The system maintains 7 predefined special periods as reference points:

| Period | Date Range | Key Characteristics |
|--------|-----------|-------------------|
| 2008 Financial Crisis | 2007-10 ~ 2009-03 | Systemic collapse, extreme volatility, deep drawdown |
| 2010 Euro Debt Crisis | 2010-04 ~ 2012-06 | Prolonged moderate decline, repeated interventions |
| 2015 China Crash | 2015-06 ~ 2016-02 | Sharp drop followed by rapid V-shaped rebound |
| 2018 Trade War | 2018-01 ~ 2018-12 | Policy-driven, choppy, exogenous shocks |
| 2020 COVID Crash | 2020-02 ~ 2020-06 | Black swan: fastest crash + fastest recovery |
| 2022 Rate Hike | 2022-01 ~ 2022-12 | Fed tightening, persistent bear market |
| 2023 AI Wave | 2023-01 ~ 2023-12 | Tech-driven sustained bull market |

### 4.2 Distance Calculation

1. Compute the 6-dim feature vector for the user's input period.
2. Compute the 6-dim feature vector for each of the 7 reference periods.
3. Stack all 8 vectors (7 reference + 1 input) and apply **joint z-score normalization** (zero mean, unit variance per feature dimension).
4. Compute **Euclidean distance** between the normalized input vector and each reference vector.
5. Rank by distance — the closest reference period is the best match.

Joint normalization ensures that all features contribute equally regardless of their original scale.

### 4.3 Category Mapping

Each reference period maps to one of **4 regime categories**:

| Category | Mapped Periods | Characteristics |
|----------|---------------|-----------------|
| `high_vol_crash` | 2008 Financial Crisis, 2020 COVID | Extreme volatility, deep drawdown, systemic risk |
| `moderate_decline` | 2010 Euro Debt, 2018 Trade War, 2022 Rate Hike | Gradual decline, choppy, policy-sensitive |
| `v_shape_recovery` | 2015 China Crash | Sharp drop + rapid rebound, high drawdown but near-zero net trend |
| `trending_bull` | 2023 AI Wave | Sustained uptrend, moderate volatility, strong positive trend |

---

## 5. Model Selection Strategy

### 5.1 Two-Level Strategy

The system employs a **dual strategy** with automatic fallback:

**Strategy A — Historical Data** (preferred):  
If a `SpecialPeriod_Summary.csv` from prior experiments exists, the system:
1. Looks up the matched period's actual R² scores per model.
2. Filters by the current forecast horizon.
3. Excludes ARIMA (benchmark), SNN (unstable), and PatchTST (unless long-trending).
4. Selects the top-K models (default K=3) with R² > 0.
5. Assigns weights proportional to R² (with PatchTST penalized by 0.3x).

**Strategy B — Heuristic Rules** (fallback):  
If no historical data is available, the system uses expert rules derived from comprehensive experiments on both SP500 and HSI markets.

### 5.2 Horizon-Aware Model Selection

Different forecast horizons require different model architectures. The system divides horizons into three bands:

| Band | Horizon Range | Rationale |
|------|:------------:|-----------|
| **Short** | H ≤ 1 | Next-day prediction; recent patterns dominate |
| **Medium** | H = 2~5 | Short-term trend following; balanced complexity |
| **Long** | H ≥ 6 | Multi-day ahead; simpler models generalize better |

**Each horizon band has its own model selection and weight assignment**, meaning a single Smart Predict run with H=1 and H=10 will use **different model ensembles** for each.

### 5.3 Market-Specific Rules

SP500 and HSI exhibit fundamentally different prediction characteristics. The system maintains **separate rule sets** for each market:

#### SP500 Rules

Based on SP500 analysis (Seed2620):
- **Overall DL ranking**: Mamba (0.64) > Bi-RNN (0.63) > Bi-GRU (0.58) > Bi-LSTM (0.48)
- Wide performance spread — weight differentiation matters
- PatchTST is catastrophic (R² = -0.80), always excluded
- SNN is unstable (R² = -0.13), always excluded

| Category | Short (H≤1) | Medium (H=2~5) | Long (H≥6) |
|----------|------------|----------------|-------------|
| `high_vol_crash` | Mamba 0.30, Bi-GRU 0.25, Bi-RNN 0.25, Bi-LSTM 0.20 | Bi-RNN 0.30, Mamba 0.25, Bi-LSTM 0.25, Bi-GRU 0.20 | Mamba 0.40, Bi-RNN 0.30, Bi-GRU 0.30 |
| `moderate_decline` | Mamba 0.30, Bi-GRU 0.25, Bi-RNN 0.25, Bi-LSTM 0.20 | Mamba 0.30, Bi-RNN 0.25, Bi-LSTM 0.25, Bi-GRU 0.20 | Mamba 0.40, Bi-RNN 0.30, Bi-GRU 0.30 |
| `v_shape_recovery` | Mamba 0.35, Bi-GRU 0.25, Bi-RNN 0.20, Bi-LSTM 0.20 | Bi-GRU 0.30, Bi-LSTM 0.25, Mamba 0.25, Bi-RNN 0.20 | Mamba 0.40, Bi-RNN 0.30, Bi-GRU 0.30 |
| `trending_bull` | Mamba 0.40, Bi-GRU 0.35, Bi-RNN 0.25 | Bi-GRU 0.35, Mamba 0.35, Bi-RNN 0.30 | Mamba 0.40, Bi-GRU 0.30, Bi-RNN 0.30 |

**Key SP500 patterns**:
- Mamba dominates Short horizon (led in 4/7 periods at H=1)
- Mamba leads Long horizon across all categories
- Bi-GRU is strong in trending/recovery markets

#### HSI Rules

Based on HSI analysis (Seed2620):
- **Overall DL ranking**: Bi-GRU (0.81) > Bi-RNN (0.81) > Mamba (0.80) > Bi-LSTM (0.80)
- Extremely tight top-4 cluster (only 0.01 R² spread) — weights are more uniform
- PatchTST is positive (R² = 0.52) but still well behind top-4
- SNN is positive (R² = 0.36) but too unstable

| Category | Short (H≤1) | Medium (H=2~5) | Long (H≥6) |
|----------|------------|----------------|-------------|
| `high_vol_crash` | Bi-RNN 0.30, Bi-GRU 0.25, Bi-LSTM 0.25, Mamba 0.20 | Bi-RNN 0.30, Mamba 0.25, Bi-LSTM 0.25, Bi-GRU 0.20 | Bi-LSTM 0.40, Bi-RNN 0.30, Bi-GRU 0.30 |
| `moderate_decline` | Bi-RNN 0.30, Bi-GRU 0.30, Mamba 0.20, Bi-LSTM 0.20 | Bi-RNN 0.30, Bi-GRU 0.28, Mamba 0.22, Bi-LSTM 0.20 | Bi-GRU 0.35, Bi-RNN 0.35, Mamba 0.30 |
| `v_shape_recovery` | Bi-GRU 0.30, Mamba 0.25, Bi-LSTM 0.25, Bi-RNN 0.20 | Bi-LSTM 0.30, Bi-GRU 0.25, Mamba 0.25, Bi-RNN 0.20 | Bi-GRU 0.40, Bi-RNN 0.30, Bi-LSTM 0.30 |
| `trending_bull` | Mamba 0.30, Bi-GRU 0.25, Bi-LSTM 0.25, Bi-RNN 0.20 | Bi-GRU 0.28, Mamba 0.28, Bi-LSTM 0.22, Bi-RNN 0.22 | Mamba 0.35, Bi-GRU 0.35, Bi-RNN 0.30 |

**Key HSI patterns**:
- Bi-RNN and Bi-GRU dominate Short horizon (led in 5/7 periods at H=1) — opposite of SP500
- Bi-LSTM is much stronger on HSI than SP500 (+0.32 R² improvement)
- Mamba only leads in `trending_bull` — unlike its broad SP500 dominance
- Weights are more uniform due to the tight performance cluster

### 5.4 PatchTST Inclusion Rule

PatchTST is **excluded by default** due to poor SP500 performance (R² = -0.80). However, it is **re-added** when:

1. The input period spans **> 3 years**, AND
2. The matched category is a **trending** type (`trending_bull`, `high_vol_crash`, or `moderate_decline`)

When included, PatchTST receives a fixed weight of **0.20**, and all other weights are re-normalized.

**Rationale**: PatchTST's patch-based attention mechanism can capture long-range patterns in sustained directional markets. On HSI it achieves positive R² (0.52), and even on SP500 it shows marginal positive R² during the 2023 AI Wave (a strong trending period). For short or volatile periods, its fine-grained temporal resolution loss is fatal.

### 5.5 Permanently Excluded Models

| Model | Reason |
|-------|--------|
| **ARIMA** | Statistical benchmark only; should not participate in ensemble prediction |
| **SNN** | Extreme instability (SP500 R² range: -1.51 to 0.56; HSI std = 0.33), not reliable |

---

## 6. Ensemble Method

### 6.1 Weighted Average Ensemble

The final prediction is a **weighted average** of individual model predictions:

$$\hat{y}_{ensemble} = \frac{\sum_{i=1}^{K} w_i \cdot \hat{y}_i}{\sum_{i=1}^{K} w_i}$$

Where:
- $w_i$ = weight assigned to model $i$ (from heuristic rules or historical R²)
- $\hat{y}_i$ = prediction from model $i$
- $K$ = number of selected models (typically 3–4)

### 6.2 Why Weighted > Equal-Weight

The original Classic Mode uses a naive equal-weight ensemble across all models, which **drags performance down** when weak models (PatchTST, SNN) are included. The SP500 experiment showed the Ensemble (R² = 0.57) underperforming both Mamba (0.64) and Bi-RNN (0.63) due to this effect.

Smart Prediction solves this by:
1. **Excluding** known poor performers entirely.
2. **Weighting** models proportionally to their expected performance.
3. **Adapting** weights per horizon and per market.

---

## 7. Walk-Forward Training Protocol

Smart Prediction uses the same walk-forward methodology as Classic Mode:

```
|<------- Training Data ------->|<-- Period -->|
   All data before period start   User's input period

Training set:  [data_start, period_start)
Validation:    Last 10% of training (for early stopping)
Test set:      Period data (with seq_len lookback)
```

- Models are trained **only on data before the period starts** — no future data leakage.
- Early stopping with patience=15 prevents overfitting.
- Each (window_size, horizon) combination trains fresh models independently.

---

## 8. Usage

### 8.1 Running Smart Predict

```bash
python SpecialPeriod.py
```

Select **Mode 2** when prompted:

```
========== Mode Selection ==========
  1. Classic Mode (manual selection of models/periods)
  2. Smart Predict (auto-match period & select best models)
====================================
Enter mode (1 or 2): 2
```

Then provide:
1. Market (SP500 or HSI)
2. Start date (YYYY-MM-DD)
3. End date (YYYY-MM-DD)
4. Sequence lengths (window sizes)
5. Forecast horizons

### 8.2 Output

The system outputs:
- **Feature vector**: 6-dim statistics of the input period
- **Similarity ranking**: Distance to all 7 reference periods
- **Best match**: Closest period and its category
- **Model selection**: Per-horizon model list with weights and strategy used
- **Prediction results**: MAE, R², RMSE for each individual model and the weighted ensemble
- **Prediction curves**: Actual vs predicted plots for each model and the ensemble
- **Summary CSV**: `SmartPredict_Summary.csv` with all results

### 8.3 Example Output

```
[FEATURES] Input period feature vector:
  MeanReturn     : -0.001234
  Volatility     : 0.023456
  MaxDrawdown    : 0.187654
  Skewness       : -0.543210
  Kurtosis       : +2.345678
  TrendStrength  : -1.234567

[MATCH] Period similarity ranking:
  1. 2020_COVID_Crash              distance=1.2345 <<<
  2. 2008_Financial_Crisis         distance=2.3456
  3. 2015_China_Crash              distance=3.4567
  ...

[RESULT] Best match: 2020_COVID_Crash (distance=1.2345)

  --- W=60 | H=5 ---
  [MODELS] Strategy: heuristic | Horizon band: medium
    Bi-RNN      : weight=0.300
    Mamba       : weight=0.250
    Bi-LSTM     : weight=0.250
    Bi-GRU      : weight=0.200
    Bi-RNN      : MAE=0.0123 | R2=0.8456
    Mamba       : MAE=0.0134 | R2=0.8234
    ...
    Ensemble    : MAE=0.0118 | R2=0.8567 (weighted)
```

---

## 9. Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| 6-dim features instead of raw prices | Invariant to price level; captures regime characteristics (volatility, trend, tail risk) |
| Joint z-score normalization | Ensures all features contribute equally to distance; prevents volatility from dominating |
| Euclidean distance over cosine similarity | We care about magnitude differences (a period with 2x volatility IS different), not just directional similarity |
| 4 categories not 7 | Similar periods share optimal model configurations; grouping reduces overfitting to specific historical episodes |
| Per-horizon model selection | The SP500 analysis showed R² drops 83% from H=1 to H=10, with completely different best models at each horizon |
| Separate SP500/HSI rules | HSI's top-4 DL models are within 0.01 R² of each other, while SP500 has 0.16 spread — requiring fundamentally different weighting strategies |
| PatchTST >3yr threshold | PatchTST needs long-range patterns to function; short volatile periods destroy its patch-based attention |
| Weighted over equal ensemble | Prevents weak models from dragging down the ensemble; the Classic Mode's Ensemble underperformed individual models due to PatchTST/SNN contamination |

---

*Report generated for SpecialPeriod.py Smart Prediction Module.*
