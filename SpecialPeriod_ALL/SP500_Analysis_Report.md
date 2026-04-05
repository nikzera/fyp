# SP500 Special Period Model Performance Analysis

> **Experiment**: SpecialPeriod_Seed2620  
> **Market**: S&P 500 (^GSPC)  
> **Models**: ARIMA, Bi-RNN, Bi-GRU, Bi-LSTM, PatchTST, Mamba, SNN, Ensemble  
> **Window Sizes**: 30, 60, 90, 120, 252  
> **Prediction Horizons**: 1-day, 5-day, 10-day  
> **Evaluation Metrics**: R², RMSE, MAE

---

## 1. Overall Performance Summary

### 1.1 Period Difficulty Ranking (Average R² across all models)

| Rank | Period | Avg R² | Avg RMSE | Avg MAE | Samples | Difficulty |
|:----:|--------|-------:|----------:|--------:|--------:|:----------:|
| 1 | 2008 Financial Crisis | **0.7268** | 0.0191 | 0.0144 | 378 | Easiest |
| 2 | 2010 Euro Debt Crisis | 0.5395 | 0.0096 | 0.0077 | 568 | |
| 3 | 2023 AI Wave | 0.3604 | 0.0277 | 0.0235 | 250 | |
| 4 | 2022 Rate Hike | 0.2723 | 0.0381 | 0.0321 | 251 | |
| 5 | 2015 China Crash | 0.2145 | 0.0117 | 0.0091 | 189 | |
| 6 | 2018 Trade War | 0.1983 | 0.0155 | 0.0119 | 251 | |
| 7 | 2020 COVID Crash | **0.0683** | 0.0438 | 0.0318 | 104 | Hardest |

**Interpretation**:

- **2008 Financial Crisis (R²=0.73)**: Despite the extreme volatility, the crisis followed a relatively clear downward-then-recovery trend driven by identifiable macro events (subprime collapse, TARP, etc.). The sustained directional movement made it structurally predictable for time-series models. The large sample size (378 trading days) also provided ample training/test signal.
- **2010 Euro Debt Crisis (R²=0.54)**: This was a prolonged, multi-phase crisis with identifiable structural breaks (Greek bailout, ECB interventions). The extended duration (568 samples) helped models learn meaningful patterns, and the relatively low RMSE (0.0096) suggests the volatility amplitude was moderate.
- **2023 AI Wave (R²=0.36)**: A trending bull market driven by the AI/LLM hype cycle. While the general upward trend was capturable, sudden sector rotations and news-driven jumps (earnings surprises from NVIDIA, Microsoft, etc.) introduced noise that limited model accuracy.
- **2022 Rate Hike (R²=0.27)**: The aggressive Federal Reserve tightening cycle created a persistent bear market, but with frequent policy-sensitive reversals. High RMSE (0.038) reflects the large absolute price swings during this volatile period.
- **2015 China Crash (R²=0.21)**: The global spillover from China's stock market crash affected SP500 through contagion channels, but the transmission was indirect and intermittent, making it harder for purely historical-pattern-based models to capture. The smallest sample size among non-COVID periods (189) further limited model learning.
- **2018 Trade War (R²=0.20)**: The US-China trade war was driven by unpredictable political decisions (tariff announcements via Twitter), creating regime shifts that are fundamentally incompatible with statistical extrapolation.
- **2020 COVID Crash (R²=0.07)**: The most difficult period by far. The pandemic caused an unprecedented speed of decline (34% drop in 23 trading days) followed by an equally rapid V-shaped recovery. With only 104 samples and extreme non-stationarity, all models essentially failed. The RMSE (0.044) was the highest across all periods, reflecting massive prediction errors.

### 1.2 Model Overall Ranking (Average R² across all conditions)

| Rank | Model | Avg R² | Avg RMSE | Avg MAE | Std(R²) | Stability |
|:----:|-------|-------:|---------:|--------:|--------:|:---------:|
| 1 | **ARIMA** | **0.7501** | 0.0146 | 0.0112 | 0.1645 | Most Stable |
| 2 | Mamba | 0.6367 | 0.0177 | 0.0137 | 0.2937 | |
| 3 | Bi-RNN | 0.6303 | 0.0187 | 0.0145 | 0.1724 | Stable |
| 4 | Bi-GRU | 0.5827 | 0.0189 | 0.0141 | 0.2517 | |
| 5 | Ensemble | 0.5687 | 0.0211 | 0.0161 | 0.1906 | |
| 6 | Bi-LSTM | 0.4798 | 0.0211 | 0.0163 | 0.3632 | |
| 7 | SNN | -0.1282 | 0.0342 | 0.0279 | 0.6969 | Unstable |
| 8 | PatchTST | **-0.7999** | 0.0427 | 0.0355 | 0.7439 | Most Unstable |

**Interpretation**:

- **ARIMA dominance**: ARIMA achieved the highest R² in every single period. This is expected because ARIMA directly models the autocorrelation structure of time series. For short-horizon predictions (H=1), the latest price is the strongest predictor, and ARIMA's autoregressive nature exploits this efficiently. Its low standard deviation (0.16) across periods indicates robustness to different market regimes.
- **Mamba as best DL model**: Mamba's selective state space mechanism allows it to dynamically focus on relevant historical information while forgetting noise. This is analogous to how financial markets exhibit both long-range dependencies (macro trends) and short-range patterns (momentum), which Mamba's architecture is designed to handle.
- **Bi-RNN's surprising stability**: Despite being the simplest recurrent architecture, Bi-RNN achieved the second-lowest std(R²) = 0.17. Its simplicity may serve as regularization, preventing overfitting to specific market regimes.
- **Bi-LSTM underperforming Bi-GRU/Bi-RNN**: LSTM's more complex gating mechanism (3 gates vs GRU's 2) may overfit in the limited special-period data, especially for longer horizons.
- **PatchTST failure**: PatchTST was designed for long-range forecasting in multivariate settings. The negative R² across most periods suggests it fails to capture the short-range dependencies critical for stock prediction. Its patch-based tokenization may lose fine-grained temporal resolution needed for financial data.
- **SNN instability**: The Spiking Neural Network's binary spike-based computation introduces quantization noise that severely degrades prediction quality. Its extreme variance (range: -1.51 to 0.56) makes it unreliable.
- **Ensemble underperformance**: The Ensemble averages all models including PatchTST and SNN, whose severely negative R² values drag down the aggregate. This demonstrates that naive ensembling can be harmful when constituent models include poorly performing ones.

---

## 2. Analysis by Prediction Horizon

### 2.1 Horizon Effect on R²

| Horizon | Avg R² | Avg RMSE | Avg MAE |
|:-------:|-------:|---------:|--------:|
| **1-day** | **0.5749** | 0.0180 | 0.0142 |
| 5-day | 0.3499 | 0.0238 | 0.0188 |
| 10-day | 0.0953 | 0.0291 | 0.0229 |

### 2.2 R² by Period x Horizon

| Period | H=1 | H=5 | H=10 | Degradation (H=1 to H=10) |
|--------|----:|----:|-----:|:--------------------------:|
| 2008 Financial Crisis | 0.7563 | 0.7058 | 0.7182 | -5.0% (minimal) |
| 2010 Euro Debt Crisis | 0.6617 | 0.4567 | 0.5002 | -24.4% |
| 2023 AI Wave | 0.4899 | 0.4136 | 0.1778 | -63.7% |
| 2022 Rate Hike | 0.5399 | 0.3284 | -0.0514 | -109.5% (collapse) |
| 2015 China Crash | 0.4756 | 0.0922 | 0.0758 | -84.1% |
| 2018 Trade War | 0.5974 | 0.2842 | -0.2868 | -148.0% (collapse) |
| 2020 COVID Crash | 0.5034 | 0.1684 | -0.4668 | -192.7% (collapse) |

**Interpretation**:

- **Horizon is the single most important factor**: R² drops from 0.57 (H=1) to 0.10 (H=10), a decline of 83%. This follows the established finding in financial econometrics that prediction accuracy decays rapidly with forecast horizon due to error accumulation and increasing uncertainty.
- **2008 Crisis is uniquely horizon-resilient**: R² barely changes from H=1 (0.76) to H=10 (0.72). This is because the crisis had strong persistent trends: once the crash began, the momentum carried through multi-day windows. The consistent directional movement meant that even 10-day-ahead predictions could leverage the established trend.
- **2020 COVID and 2018 Trade War collapse at H=10**: Both have negative R² at H=10 (meaning worse than a simple mean predictor). The 2020 COVID crash's V-shaped recovery and the 2018 trade war's tweet-driven reversals created conditions where 10-day-ahead predictions were directionally wrong more often than right.
- **2023 AI Wave degrades significantly**: While H=1 (0.49) was moderate, H=10 drops to 0.18. The AI boom had micro-level rotations (sector shifts between AI chipmakers, software, and cloud providers) that were predictable short-term but not medium-term.

### 2.3 Best Deep Learning Model per Period x Horizon

| Period | H=1 | H=5 | H=10 |
|--------|-----|-----|------|
| 2008 Financial Crisis | Bi-GRU (0.9864) | Bi-LSTM (0.9489) | Mamba (0.9019) |
| 2010 Euro Debt Crisis | Bi-GRU (0.9639) | Bi-LSTM (0.8729) | Bi-RNN (0.8301) |
| 2015 China Crash | Mamba (0.8997) | Bi-GRU (0.5739) | Bi-RNN (0.3939) |
| 2018 Trade War | Mamba (0.8744) | Mamba (0.4903) | Bi-RNN (0.2654) |
| 2020 COVID Crash | Bi-RNN (0.7639) | Bi-RNN (0.5068) | Bi-RNN (-0.1485) |
| 2022 Rate Hike | Mamba (0.9042) | Mamba (0.7593) | Mamba (0.4949) |
| 2023 AI Wave | Mamba (0.9561) | Bi-GRU (0.8339) | Bi-GRU (0.6662) |

**Interpretation**:

- **H=1 is dominated by Mamba (4/7 periods)**: Mamba's state-space mechanism excels at capturing short-range dependencies, making it the best 1-day predictor in most market regimes.
- **H=5 shows mixed leadership**: No single model dominates, suggesting that medium-term prediction requires different inductive biases depending on market conditions.
- **H=10 favors Bi-RNN (4/7 periods)**: For longer horizons, simpler architectures appear to generalize better. Bi-RNN's minimal parameterization acts as an implicit regularizer, preventing the overfitting that afflicts more complex models at longer horizons.
- **Mamba dominates the 2022 Rate Hike across all horizons**: The Federal Reserve's rate-hike cycle had a strong structural signal (scheduled FOMC meetings, forward guidance) that Mamba's selective state space could effectively model as a long-range dependency.

---

## 3. Analysis by Window Size

### 3.1 Window Size Effect on R²

| Window | Avg R² | Avg RMSE |
|:------:|-------:|---------:|
| 30 | 0.3769 | 0.0224 |
| 60 | 0.3585 | 0.0237 |
| 90 | 0.3218 | 0.0239 |
| **120** | **0.3829** | 0.0229 |
| 252 | 0.2600 | 0.0253 |

### 3.2 R² by Window x Horizon (averaged across all models and periods)

| Window \ Horizon | H=1 | H=5 | H=10 |
|:----------------:|----:|----:|-----:|
| 30 | 0.6310 | 0.3430 | 0.1566 |
| 60 | 0.5941 | 0.3347 | 0.1466 |
| 90 | 0.5273 | 0.3085 | 0.1298 |
| **120** | 0.5937 | **0.4158** | 0.1391 |
| 252 | 0.5282 | 0.3475 | **-0.0956** |

**Interpretation**:

- **Window size has a relatively minor effect** compared to horizon and model choice. The best (W=120, R²=0.38) and worst (W=252, R²=0.26) differ by only 0.12.
- **W=120 (approximately 6 months) is optimal overall**: This strikes a balance between having enough historical context to capture market trends and avoiding the inclusion of stale, irrelevant data. Six months roughly corresponds to two earnings cycles, capturing meaningful fundamental signals.
- **W=30 is competitive for H=1 (R²=0.63)**: For next-day prediction, recent data is most informative. A 30-day window captures the current market microstructure without noise from older, potentially irrelevant data.
- **W=252 (1 year) is worst, especially at H=10 (R²=-0.10)**: A full year of lookback introduces older data that may reflect different market regimes, confusing the model. This is particularly problematic for longer horizons where error accumulation is already high.
- **W=90 slightly underperforms**: This may correspond to an awkward in-between where the window captures partial quarterly cycles but not complete ones, introducing noise from incomplete earnings periods.

### 3.3 Window Size Effect per Model

| Model \ Window | W=30 | W=60 | W=90 | W=120 | W=252 |
|:--------------:|-----:|-----:|-----:|------:|------:|
| ARIMA | 0.7362 | 0.7492 | 0.7507 | 0.7569 | 0.7576 |
| Bi-RNN | 0.6821 | 0.6882 | 0.6493 | 0.6668 | 0.4651 |
| Bi-GRU | 0.6384 | 0.6127 | 0.6269 | 0.6046 | 0.4312 |
| Bi-LSTM | 0.5495 | 0.4797 | 0.2776 | 0.5377 | 0.5542 |
| Mamba | 0.6413 | 0.5861 | 0.6742 | 0.6737 | 0.6082 |
| PatchTST | -0.6740 | -0.6510 | -0.6579 | -0.7952 | -1.0214 |
| SNN | -0.0092 | -0.1731 | -0.3115 | -0.0154 | -0.4455 |
| Ensemble | 0.5840 | 0.5760 | 0.5655 | 0.5868 | 0.5329 |

**Interpretation**:

- **ARIMA improves monotonically with window size**: More data helps ARIMA estimate autoregressive coefficients more accurately, as it relies on statistical parameter estimation.
- **Deep learning models generally prefer shorter windows (W=30~120)**:
  - **Bi-RNN**: Sharp degradation at W=252 (0.47 vs 0.68 at W=30), suggesting it cannot effectively filter relevant signals from a full year of data.
  - **Bi-GRU**: Similar pattern to Bi-RNN, with W=30 being optimal.
  - **Mamba**: Peaks at W=90-120, consistent with its state-space architecture's ability to selectively retain longer-range patterns without being overwhelmed by noise.
- **PatchTST gets progressively worse with larger windows**: Its patch-based approach creates increasingly many tokens with larger windows, potentially causing attention dilution.

---

## 4. Detailed Period-by-Period Analysis

### 4.1 2008 Financial Crisis (2007-10 to 2009-03)

| Model | H=1 | H=5 | H=10 | Avg |
|-------|----:|----:|-----:|----:|
| ARIMA | 0.9902 | 0.9676 | 0.9394 | **0.9657** |
| Bi-GRU | 0.9864 | 0.9483 | 0.8967 | 0.9438 |
| Bi-LSTM | 0.9730 | 0.9489 | 0.9006 | 0.9409 |
| Mamba | 0.9777 | 0.9333 | 0.9019 | 0.9376 |
| Bi-RNN | 0.6537 | 0.9354 | 0.5878 | 0.7256 |
| Ensemble | 0.8782 | 0.8382 | 0.8047 | 0.8404 |
| SNN | 0.4154 | 0.7586 | 0.4975 | 0.5572 |
| PatchTST | 0.1760 | -0.6839 | 0.2173 | -0.0968 |

**Analysis**: All mainstream models achieve excellent performance (R² > 0.90 for ARIMA, Bi-GRU, Bi-LSTM, Mamba). The 2008 crisis featured a strong, persistent downtrend with high autocorrelation, making it ideal for time-series models. The large sample size (378) and sustained directional trend allowed models to learn the dominant pattern. Notably, Bi-RNN has an anomalously low H=1 score (0.65), likely due to its inability to capture the complex intra-day volatility clustering during the peak of the crisis, while performing well at H=5 (0.94) where the dominant trend reasserts.

### 4.2 2010 Euro Debt Crisis (2010-04 to 2012-06)

| Model | H=1 | H=5 | H=10 | Avg |
|-------|----:|----:|-----:|----:|
| ARIMA | 0.9744 | 0.8831 | 0.8112 | **0.8896** |
| Mamba | 0.9601 | 0.8684 | 0.8054 | 0.8780 |
| Bi-RNN | 0.9559 | 0.8708 | 0.8301 | 0.8856 |
| Bi-LSTM | 0.9633 | 0.8729 | 0.7900 | 0.8754 |
| Bi-GRU | 0.9639 | 0.4772 | 0.5143 | 0.6518 |
| Ensemble | 0.8544 | 0.7344 | 0.7102 | 0.7663 |
| SNN | 0.1308 | 0.5713 | 0.4405 | 0.3809 |
| PatchTST | -0.5093 | -1.6243 | -0.9004 | -1.0113 |

**Analysis**: The Euro Debt Crisis had the largest sample size (568 days) and relatively moderate volatility (lowest RMSE among all periods at 0.0096), creating favorable learning conditions. Most deep learning models performed comparably at H=1 (all around 0.96). The anomaly is Bi-GRU's sharp drop at H=5 (0.48), suggesting potential overfitting to short-range patterns that didn't generalize to medium-term predictions in this specific period.

### 4.3 2015 China Crash (2015-06 to 2016-02)

| Model | H=1 | H=5 | H=10 | Avg |
|-------|----:|----:|-----:|----:|
| ARIMA | 0.9130 | 0.5869 | 0.3015 | **0.6005** |
| Bi-RNN | 0.8305 | 0.4766 | 0.3939 | 0.5670 |
| Bi-GRU | 0.8613 | 0.5739 | 0.2302 | 0.5551 |
| Bi-LSTM | 0.8240 | 0.5588 | 0.2032 | 0.5287 |
| Mamba | 0.8997 | 0.5548 | 0.0807 | 0.5117 |
| Ensemble | 0.7533 | 0.5128 | 0.2649 | 0.5103 |
| SNN | -0.0419 | -0.0079 | -0.0766 | -0.0422 |
| PatchTST | -1.2353 | -2.5181 | -0.7911 | -1.5148 |

**Analysis**: Performance degrades significantly compared to the earlier crises. The China Crash's impact on SP500 was transmitted through complex cross-market contagion channels (capital flows, commodity prices, currency markets) rather than domestic fundamentals. This indirect transmission created noisy, non-stationary patterns. The small sample size (189) exacerbated the problem. Mamba excels at H=1 (0.90) due to its ability to model the rapid contagion dynamics, but collapses at H=10 (0.08) as these dynamics become unpredictable over longer horizons.

### 4.4 2018 Trade War (2018-01 to 2018-12)

| Model | H=1 | H=5 | H=10 | Avg |
|-------|----:|----:|-----:|----:|
| ARIMA | 0.9084 | 0.5851 | 0.2500 | **0.5812** |
| Mamba | 0.8744 | 0.4903 | 0.1470 | 0.5039 |
| Bi-RNN | 0.7591 | 0.4131 | 0.2654 | 0.4792 |
| Bi-GRU | 0.8089 | 0.3991 | -0.1310 | 0.3590 |
| Ensemble | 0.7638 | 0.4478 | 0.0899 | 0.4338 |
| SNN | 0.3853 | -0.0209 | -0.5355 | -0.0570 |
| Bi-LSTM | 0.7228 | 0.2963 | -1.3358 | -0.1056 |
| PatchTST | -0.4435 | -0.3369 | -1.0446 | -0.6083 |

**Analysis**: The trade war was driven by exogenous political shocks (tariff announcements, trade negotiations) that are fundamentally unpredictable from historical price data alone. This represents a textbook case of "news-driven" markets where technical models struggle. Bi-LSTM's catastrophic H=10 performance (-1.34) suggests severe overfitting to short-term patterns that reverse over 10-day windows as new trade-war developments altered market direction. Mamba performs best among DL models, possibly because its selective gating can quickly adapt to regime changes.

### 4.5 2020 COVID Crash (2020-02 to 2020-06)

| Model | H=1 | H=5 | H=10 | Avg |
|-------|----:|----:|-----:|----:|
| ARIMA | 0.8896 | 0.6057 | 0.2216 | **0.5723** |
| Bi-RNN | 0.7639 | 0.5068 | -0.1485 | 0.3740 |
| Ensemble | 0.6961 | 0.3996 | -0.2448 | 0.2836 |
| Bi-LSTM | 0.6573 | 0.3490 | -0.3436 | 0.2209 |
| Bi-GRU | 0.7132 | 0.3027 | -0.3812 | 0.2116 |
| Mamba | 0.6830 | 0.3249 | -0.7330 | 0.0917 |
| SNN | -0.1103 | -0.2828 | -1.0407 | -0.4779 |
| PatchTST | -0.2655 | -0.8587 | -1.0644 | -0.7295 |

**Analysis**: The most challenging period for all models. COVID-19 caused an unprecedented "black swan" event: the fastest bear market in history (34% decline in 23 days) followed by an equally unprecedented recovery fueled by massive fiscal/monetary stimulus. With only 104 samples and extreme non-stationarity, models trained on historical data had no comparable patterns to reference. At H=10, every DL model has negative R², meaning they are all worse than simply predicting the mean. Mamba, despite being the best DL model overall, particularly struggles here (-0.73 at H=10), suggesting that its state-space mechanism memorized pre-crash dynamics that became completely irrelevant during the recovery phase.

### 4.6 2022 Rate Hike (2022-01 to 2022-12)

| Model | H=1 | H=5 | H=10 | Avg |
|-------|----:|----:|-----:|----:|
| ARIMA | 0.9512 | 0.7861 | 0.5941 | **0.7771** |
| Mamba | 0.9042 | 0.7593 | 0.4949 | 0.7195 |
| Bi-RNN | 0.8878 | 0.5750 | 0.4695 | 0.6441 |
| Ensemble | 0.8407 | 0.6587 | 0.3216 | 0.6070 |
| Bi-GRU | 0.8590 | 0.6143 | 0.1506 | 0.5413 |
| Bi-LSTM | 0.7627 | 0.5892 | 0.1882 | 0.5133 |
| SNN | 0.3362 | 0.2319 | 0.1835 | 0.2505 |
| PatchTST | -1.2229 | -1.5877 | -2.8135 | -1.8747 |

**Analysis**: The 2022 rate-hike cycle created a structurally driven bear market where market movements were tightly coupled to Federal Reserve communications and economic data releases (CPI, employment). This made the market more "regime-based" — periods of sell-off following hawkish signals and relief rallies on dovish hints. Mamba excels here (best DL model with R²=0.72), likely because its state-space mechanism can model the conditional regime dynamics between Fed actions and market responses. PatchTST has its worst performance across all periods (-1.87), confirming its fundamental unsuitability for regime-sensitive financial data.

### 4.7 2023 AI Wave (2023-01 to 2023-12)

| Model | H=1 | H=5 | H=10 | Avg |
|-------|----:|----:|-----:|----:|
| ARIMA | 0.9758 | 0.8756 | 0.7422 | **0.8645** |
| Bi-GRU | 0.9495 | 0.8339 | 0.6662 | 0.8165 |
| Mamba | 0.9561 | 0.8327 | 0.6554 | 0.8147 |
| Bi-RNN | 0.8699 | 0.7876 | 0.5523 | 0.7366 |
| Ensemble | 0.7286 | 0.5704 | 0.3184 | 0.5391 |
| Bi-LSTM | 0.8619 | 0.5602 | -0.2682 | 0.3847 |
| PatchTST | 0.3778 | -0.0192 | 0.3503 | 0.2363 |
| SNN | -1.8006 | -1.1323 | -1.5945 | -1.5091 |

**Analysis**: The 2023 AI-driven bull market proved relatively predictable for most models. The strong upward trend, driven by consistent AI investment narratives, created favorable conditions for trend-following models. This is the only period where PatchTST achieves positive R² (0.24), suggesting it can capture longer-range trending patterns when the market has a clear directional bias. However, SNN has its worst overall performance here (-1.51), indicating that binary spike-based computation fundamentally cannot represent the smooth, sustained uptrend characteristic of the AI boom.

---

## 5. Cross-Cutting Insights

### 5.1 Trend-Driven vs. Event-Driven Periods

The results reveal a clear dichotomy:

| Category | Periods | Avg R² | Characteristics |
|----------|---------|-------:|-----------------|
| **Trend-driven** | 2008 Crisis, 2010 Euro Debt, 2023 AI Wave | **0.54** | Sustained directional movement, high autocorrelation |
| **Event-driven** | 2018 Trade War, 2020 COVID, 2015 China Crash | **0.16** | Exogenous shocks, regime shifts, low predictability |
| **Mixed** | 2022 Rate Hike | 0.27 | Policy-driven with structural breaks |

Models perform approximately **3.4x better** (by R²) in trend-driven periods. This aligns with the efficient market hypothesis: trend-driven periods have stronger serial correlation that statistical models can exploit, while event-driven periods approach random walk behavior.

### 5.2 The Horizon-Complexity Trade-off

| Model Complexity | Best at H=1 | Best at H=10 | Interpretation |
|:-----------------|:-----------:|:------------:|:---------------|
| Low (ARIMA, Bi-RNN) | Competitive | **Winner** | Simple models generalize better at long horizons |
| Medium (Bi-GRU, Mamba) | **Winner** | Competitive | Optimal for short-to-medium horizons |
| High (PatchTST, SNN) | Worst | Worst | Overly complex for this task |

This demonstrates the **bias-variance trade-off** in action: higher-complexity models capture more signal at short horizons but accumulate more error at long horizons due to overfitting.

### 5.3 Model Architecture Suitability Summary

| Model | Strengths | Weaknesses | Best Use Case |
|-------|-----------|------------|---------------|
| **ARIMA** | Highest accuracy overall, most stable, best at all horizons | No nonlinear modeling, not end-to-end learnable | Baseline benchmark; H=1 production use |
| **Mamba** | Best DL for H=1, excels in regime-driven markets (2022, 2023) | Degrades in extreme events (COVID) | Short-horizon prediction in structured markets |
| **Bi-RNN** | Most stable DL model, best DL at H=10 | Lower peak performance | Long-horizon prediction, volatile markets |
| **Bi-GRU** | Strong short-horizon, good in trending markets | Inconsistent at H=5/10 (e.g., Euro Debt anomaly) | Short-horizon in trending markets |
| **Bi-LSTM** | Good in clear crises (2008, 2010) | Severe overfitting risk at H=10 | Only in well-defined trend periods |
| **Ensemble** | Moderate and consistent | Dragged down by poor constituents | Not recommended without model selection |
| **PatchTST** | Marginal positive R² only in trending 2023 | Negative R² in 6/7 periods | Not suitable for financial time series |
| **SNN** | Occasionally useful in crisis periods | Extreme instability, mostly negative R² | Not recommended |

### 5.4 Key Takeaways

1. **ARIMA is the undisputed champion** for stock index prediction across all special periods. Its dominance suggests that the SP500's autocorrelation structure is the strongest predictive signal, which statistical methods exploit more efficiently than neural networks.

2. **Among deep learning models, Mamba and Bi-RNN form the optimal pair**: Mamba for short-horizon (H=1) prediction in normal-to-structured markets, and Bi-RNN for longer horizons (H=5, H=10) or highly volatile conditions where simplicity aids generalization.

3. **Prediction horizon matters more than any other factor**: The R² decline from H=1 (0.57) to H=10 (0.10) dwarfs the effect of model choice, window size, or market period. This suggests that improving long-horizon prediction requires fundamentally different approaches (e.g., incorporating external data, regime detection) rather than architectural improvements alone.

4. **Black swan events remain fundamentally unpredictable**: The 2020 COVID crash demonstrates that no amount of historical pattern learning can prepare models for truly unprecedented events. Risk management strategies should not rely solely on model predictions during extreme market conditions.

5. **Window size is a secondary concern**: W=120 is marginally optimal, but the difference between windows is small. Practitioners should prioritize model selection and horizon calibration over window tuning.

---

*Report generated from SpecialPeriod_Seed2620 experiment data.*
