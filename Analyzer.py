# Analyzer.py — Results Analysis & Comparison Tool
# Analyzes SpecialPeriod result folders.
# Generates comparison tables, degradation analysis, and publication-ready charts.

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import sys

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. Auto-discover result folders
# ==========================================
def discover_folders():
    """Find all SpecialPeriod result folders."""
    sp_folders = sorted(glob.glob(os.path.join(BASE_DIR, "SpecialPeriod_Seed*")))

    all_runs = []
    for f in sp_folders:
        csv_path = os.path.join(f, "SpecialPeriod_Summary.csv")
        if os.path.exists(csv_path):
            seed = os.path.basename(f).replace("SpecialPeriod_Seed", "")
            all_runs.append({'seed': seed, 'folder': f, 'csv': csv_path})

    return all_runs

def load_run(run_info):
    """Load CSV and normalize column names."""
    df = pd.read_csv(run_info['csv'])
    df['Seed'] = run_info['seed']
    if 'RMSE' not in df.columns:
        df['RMSE'] = np.nan
    if 'N_samples' not in df.columns:
        df['N_samples'] = np.nan
    return df

# ==========================================
# 2. Interactive selection
# ==========================================
def select_runs(all_runs):
    print("\n========== Available Runs ==========")
    for i, r in enumerate(all_runs):
        config_path = os.path.join(r['folder'], 'run_config.txt')
        extra = ""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
                extra = " | ".join(lines[1:4])  # Models, Markets, Periods
        print(f"  {i + 1}. Seed={r['seed']}  {extra}")
    print(f"  0. Load ALL runs")
    print("====================================")

    user_input = input("Enter run numbers separated by commas (e.g. 1,3) or 0 for all: ").strip()
    if user_input == '0' or user_input == '':
        return all_runs

    selected = []
    for part in user_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(all_runs):
                selected.append(all_runs[idx])
    return selected if selected else all_runs

# ==========================================
# 3. Analysis Functions
# ==========================================

def print_divider(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def analyze_overview(df, out_folder):
    """Table 1: Overall performance summary per model."""
    print_divider("1. OVERALL PERFORMANCE SUMMARY")

    # Group by Model, Period — average across all Window/Horizon
    summary = df.groupby(['Model', 'Period']).agg(
        MAE_mean=('MAE', 'mean'),
        MAE_std=('MAE', 'std'),
        R2_mean=('R2', 'mean'),
        R2_std=('R2', 'std'),
        Count=('MAE', 'count')
    ).round(4)

    print(summary.to_string())
    summary.to_csv(os.path.join(out_folder, "01_overall_summary.csv"))

    # Flag: best and worst model per period
    print("\n  [HIGHLIGHT] Best / Worst model per period (by mean R²):")
    for period in df['Period'].unique():
        sub = df[df['Period'] == period].groupby('Model')['R2'].mean()
        if len(sub) < 2:
            continue
        best = sub.idxmax()
        worst = sub.idxmin()
        print(f"    {period}: BEST = {best} (R²={sub[best]:.4f}) | WORST = {worst} (R²={sub[worst]:.4f})")

def analyze_model_ranking(df, out_folder):
    """Table 2: Rank models for each (Period, Window, Horizon) combo."""
    print_divider("2. MODEL RANKING BY CONDITION")

    rankings = []
    for (period, win, h), grp in df.groupby(['Period', 'Window', 'Horizon']):
        grp_sorted = grp.sort_values('R2', ascending=False).reset_index(drop=True)
        for rank, (_, row) in enumerate(grp_sorted.iterrows(), 1):
            rankings.append({
                'Period': period, 'Window': win, 'Horizon': h,
                'Rank': rank, 'Model': row['Model'],
                'R2': round(row['R2'], 4), 'MAE': round(row['MAE'], 4)
            })

    df_rank = pd.DataFrame(rankings)
    df_rank.to_csv(os.path.join(out_folder, "02_model_rankings.csv"), index=False)

    # Average rank per model
    avg_rank = df_rank.groupby('Model')['Rank'].mean().sort_values()
    print("\n  Average Rank (lower = better):")
    for model, rank in avg_rank.items():
        medal = ""
        if rank == avg_rank.min():
            medal = " *** BEST ***"
        elif rank == avg_rank.max():
            medal = " (weakest)"
        print(f"    {model:12s}: {rank:.2f}{medal}")

    # Win count: how many times each model ranked #1
    wins = df_rank[df_rank['Rank'] == 1]['Model'].value_counts()
    print("\n  First-place wins:")
    for model, count in wins.items():
        print(f"    {model:12s}: {count} wins")

    return df_rank

def analyze_horizon_sensitivity(df, out_folder):
    """Table 3: How R² degrades as Horizon increases."""
    print_divider("3. HORIZON SENSITIVITY (R² decay with longer forecast)")

    if df['Horizon'].nunique() < 2:
        print("  [SKIP] Only one horizon available.")
        return

    rows = []
    for (model, period, win), grp in df.groupby(['Model', 'Period', 'Window']):
        grp = grp.sort_values('Horizon')
        if len(grp) < 2:
            continue
        h_min = grp['Horizon'].min()
        h_max = grp['Horizon'].max()
        r2_short = grp[grp['Horizon'] == h_min]['R2'].values[0]
        r2_long = grp[grp['Horizon'] == h_max]['R2'].values[0]
        if r2_short > 0:
            decay_pct = (r2_short - r2_long) / abs(r2_short) * 100
        else:
            decay_pct = np.nan
        rows.append({
            'Model': model, 'Period': period, 'Window': win,
            f'R2_H{h_min}': round(r2_short, 4),
            f'R2_H{h_max}': round(r2_long, 4),
            'R2_Decay%': round(decay_pct, 2)
        })

    df_decay = pd.DataFrame(rows)
    df_decay.to_csv(os.path.join(out_folder, "03_horizon_sensitivity.csv"), index=False)

    # Average decay per model
    avg_decay = df_decay.groupby('Model')['R2_Decay%'].mean().sort_values()
    print("\n  Average R² Decay% (H=short → H=long, lower = more robust):")
    for model, decay in avg_decay.items():
        flag = ""
        if decay == avg_decay.min():
            flag = " *** MOST STABLE ***"
        elif decay == avg_decay.max():
            flag = " (most sensitive)"
        print(f"    {model:12s}: {decay:+.2f}%{flag}")

def analyze_window_sensitivity(df, out_folder):
    """Table 4: Best window length per model."""
    print_divider("4. OPTIMAL WINDOW LENGTH PER MODEL")

    if df['Window'].nunique() < 2:
        print("  [SKIP] Only one window length available.")
        return

    best_windows = []
    for (model, period), grp in df.groupby(['Model', 'Period']):
        avg_by_win = grp.groupby('Window')['R2'].mean()
        best_win = avg_by_win.idxmax()
        best_r2 = avg_by_win.max()
        worst_win = avg_by_win.idxmin()
        worst_r2 = avg_by_win.min()
        best_windows.append({
            'Model': model, 'Period': period,
            'Best_Window': best_win, 'Best_R2': round(best_r2, 4),
            'Worst_Window': worst_win, 'Worst_R2': round(worst_r2, 4),
            'Spread': round(best_r2 - worst_r2, 4)
        })

    df_bw = pd.DataFrame(best_windows)
    df_bw.to_csv(os.path.join(out_folder, "04_optimal_windows.csv"), index=False)
    print(df_bw.to_string(index=False))

def analyze_special_flags(df, out_folder):
    """Table 5: Flag anomalies and notable patterns."""
    print_divider("5. SPECIAL FLAGS & ANOMALIES")

    flags = []

    # Flag 1: Negative R² (model worse than predicting mean)
    neg_r2 = df[df['R2'] < 0]
    if not neg_r2.empty:
        for _, row in neg_r2.iterrows():
            flags.append({
                'Flag': 'NEGATIVE_R2',
                'Severity': 'CRITICAL',
                'Model': row['Model'], 'Period': row.get('Period', ''),
                'Window': row['Window'], 'Horizon': row['Horizon'],
                'Value': f"R2={row['R2']:.4f}",
                'Note': 'Model is WORSE than predicting the mean'
            })
        print(f"\n  [CRITICAL] {len(neg_r2)} cases with NEGATIVE R² (model worse than mean):")
        for _, row in neg_r2.iterrows():
            print(f"    {row['Model']} | {row.get('Period','')} | W={row['Window']} H={row['Horizon']} → R²={row['R2']:.4f}")

    # Flag 2: Very low R² (< 0.5) when others are high
    for (period, win, h), grp in df.groupby(['Period', 'Window', 'Horizon']):
        if len(grp) < 2:
            continue
        median_r2 = grp['R2'].median()
        for _, row in grp.iterrows():
            if row['R2'] < 0.5 and median_r2 > 0.7:
                flags.append({
                    'Flag': 'OUTLIER_LOW',
                    'Severity': 'WARNING',
                    'Model': row['Model'], 'Period': period,
                    'Window': win, 'Horizon': h,
                    'Value': f"R2={row['R2']:.4f} (median={median_r2:.4f})",
                    'Note': 'Significantly below peers'
                })

    # Flag 3: Suspiciously high R² with H>1 (potential data leakage concern)
    suspect = df[(df['Horizon'] >= 10) & (df['R2'] > 0.99)]
    for _, row in suspect.iterrows():
        flags.append({
            'Flag': 'SUSPECT_HIGH_R2',
            'Severity': 'INFO',
            'Model': row['Model'], 'Period': row.get('Period', ''),
            'Window': row['Window'], 'Horizon': row['Horizon'],
            'Value': f"R2={row['R2']:.4f}",
            'Note': 'Unusually high R² for long horizon — verify no leakage'
        })

    # Flag 4: MAE > 0.05 (large error on normalized data)
    high_mae = df[df['MAE'] > 0.05]
    for _, row in high_mae.iterrows():
        flags.append({
            'Flag': 'HIGH_MAE',
            'Severity': 'WARNING',
            'Model': row['Model'], 'Period': row.get('Period', ''),
            'Window': row['Window'], 'Horizon': row['Horizon'],
            'Value': f"MAE={row['MAE']:.4f}",
            'Note': 'Error exceeds 5% of normalized range'
        })

    # Flag 5: R² drops > 50% from H=1 to H=max within same model/period/window
    for (model, period, win), grp in df.groupby(['Model', 'Period', 'Window']):
        grp = grp.sort_values('Horizon')
        if len(grp) < 2:
            continue
        r2_first = grp.iloc[0]['R2']
        r2_last = grp.iloc[-1]['R2']
        if r2_first > 0 and (r2_first - r2_last) / abs(r2_first) > 0.5:
            flags.append({
                'Flag': 'SEVERE_HORIZON_DECAY',
                'Severity': 'WARNING',
                'Model': model, 'Period': period,
                'Window': win, 'Horizon': f"{grp.iloc[0]['Horizon']}→{grp.iloc[-1]['Horizon']}",
                'Value': f"R2: {r2_first:.4f}→{r2_last:.4f} ({(r2_first-r2_last)/abs(r2_first)*100:.1f}% drop)",
                'Note': 'R² drops over 50% as horizon increases'
            })

    if flags:
        df_flags = pd.DataFrame(flags)
        df_flags.to_csv(os.path.join(out_folder, "05_special_flags.csv"), index=False)

        # Summary counts
        print(f"\n  Total flags: {len(flags)}")
        for severity in ['CRITICAL', 'WARNING', 'INFO']:
            count = len([f for f in flags if f['Severity'] == severity])
            if count > 0:
                print(f"    {severity}: {count}")
    else:
        print("  No anomalies detected.")

    return flags

def analyze_vs_arima_baseline(df, out_folder):
    """Table 6B: Compare each model's R² improvement over the ARIMA baseline."""
    print_divider("6A. IMPROVEMENT OVER ARIMA BASELINE")

    if 'ARIMA' not in df['Model'].values:
        print("  [SKIP] ARIMA baseline not found in results.")
        return

    arima_avg = df[df['Model'] == 'ARIMA'].groupby(['Period', 'Window', 'Horizon'])['R2'].mean()
    other_models = [m for m in df['Model'].unique() if m not in ('ARIMA', 'Ensemble')]

    rows = []
    for (period, win, h), grp in df[df['Model'].isin(other_models)].groupby(['Period', 'Window', 'Horizon']):
        key = (period, win, h)
        if key not in arima_avg.index:
            continue
        arima_r2 = arima_avg[key]
        for _, row in grp.iterrows():
            delta = row['R2'] - arima_r2
            rows.append({
                'Model': row['Model'], 'Period': period,
                'Window': win, 'Horizon': h,
                'Model_R2': round(row['R2'], 4),
                'ARIMA_R2': round(arima_r2, 4),
                'Delta_R2': round(delta, 4),
                'Beats_ARIMA': 'Yes' if delta > 0 else 'No'
            })

    if not rows:
        print("  [SKIP] No matching conditions to compare.")
        return

    df_vs = pd.DataFrame(rows)
    df_vs.to_csv(os.path.join(out_folder, "06a_vs_arima_baseline.csv"), index=False)

    # Summary: win rate and average improvement per model
    print("\n  Model vs ARIMA summary:")
    for model in other_models:
        sub = df_vs[df_vs['Model'] == model]
        if sub.empty:
            continue
        win_rate = (sub['Beats_ARIMA'] == 'Yes').mean() * 100
        avg_delta = sub['Delta_R2'].mean()
        print(f"    {model:12s}: Win Rate={win_rate:.1f}%  Avg Delta R²={avg_delta:+.4f}")

    # Pivot: average delta per model × period
    pivot = df_vs.pivot_table(index='Model', columns='Period', values='Delta_R2', aggfunc='mean').round(4)
    if not pivot.empty:
        print("\n  Avg R² Improvement over ARIMA (Model × Period):")
        print(pivot.to_string())
        pivot.to_csv(os.path.join(out_folder, "06a_vs_arima_pivot.csv"))


def analyze_cross_period_robustness(df, out_folder):
    """Table 6: Compare model performance across special periods (degradation analysis)."""
    print_divider("6. CROSS-PERIOD ROBUSTNESS ANALYSIS")

    periods = [p for p in df['Period'].unique() if p != 'Full_Test']
    if len(periods) < 2:
        print("  [SKIP] Need at least 2 special periods for cross-period comparison.")
        # Still show single-period stats if available
        if len(periods) == 1:
            sub = df[df['Period'] == periods[0]]
            avg = sub.groupby('Model')[['MAE', 'R2']].mean().round(4).sort_values('R2', ascending=False)
            print(f"\n  Single period ({periods[0]}) average performance:")
            print(avg.to_string())
        return

    # Average R² per model per period
    pivot = df[df['Period'] != 'Full_Test'].pivot_table(
        index='Model', columns='Period', values='R2', aggfunc='mean'
    ).round(4)

    print("\n  Mean R² per Model × Period:")
    print(pivot.to_string())
    pivot.to_csv(os.path.join(out_folder, "06_cross_period_R2.csv"))

    # Robustness = std across periods (lower = more stable)
    if pivot.shape[1] >= 2:
        pivot['Mean_R2'] = pivot.mean(axis=1)
        pivot['Std_R2'] = pivot.iloc[:, :-1].std(axis=1)
        pivot['Robustness'] = pivot['Mean_R2'] - pivot['Std_R2']  # higher = better + more stable

        print("\n  Robustness Score (Mean_R² - Std_R², higher = better + stable):")
        for model in pivot.sort_values('Robustness', ascending=False).index:
            r = pivot.loc[model]
            flag = ""
            if model == pivot['Robustness'].idxmax():
                flag = " *** MOST ROBUST ***"
            print(f"    {model:12s}: Mean={r['Mean_R2']:.4f}  Std={r['Std_R2']:.4f}  Score={r['Robustness']:.4f}{flag}")

def analyze_cross_seed(df, out_folder):
    """Table 7: If multiple seeds exist, compare consistency."""
    print_divider("7. CROSS-SEED CONSISTENCY")

    if df['Seed'].nunique() < 2:
        print("  [SKIP] Only one seed loaded. Run multiple seeds to compare consistency.")
        return

    # Average R² per model across seeds
    pivot = df.pivot_table(index='Model', columns='Seed', values='R2', aggfunc='mean').round(4)
    print("\n  Mean R² per Model × Seed:")
    print(pivot.to_string())

    pivot['Seed_Std'] = pivot.std(axis=1)
    pivot['Seed_Mean'] = pivot.mean(axis=1)

    print("\n  Cross-seed stability (lower Std = more reproducible):")
    for model in pivot.sort_values('Seed_Std').index:
        r = pivot.loc[model]
        flag = " *** MOST STABLE ***" if model == pivot['Seed_Std'].idxmin() else ""
        print(f"    {model:12s}: Mean={r['Seed_Mean']:.4f}  Std={r['Seed_Std']:.4f}{flag}")

    pivot.to_csv(os.path.join(out_folder, "07_cross_seed.csv"))

# ==========================================
# 4. Visualization
# ==========================================

def plot_r2_heatmap(df, out_folder):
    """Heatmap: Model × Period, averaged across Window/Horizon."""
    periods = df['Period'].unique()
    models = [m for m in df['Model'].unique() if m != 'Ensemble']

    if len(periods) < 1 or len(models) < 1:
        return

    pivot = df[df['Model'] != 'Ensemble'].pivot_table(
        index='Model', columns='Period', values='R2', aggfunc='mean'
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(periods) * 2), max(4, len(models) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                linewidths=0.5, ax=ax, vmin=min(0, pivot.min().min()))
    ax.set_title("Mean R² — Model × Period")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "plot_R2_heatmap.png"), dpi=150)
    plt.close()

def plot_mae_heatmap(df, out_folder):
    pivot = df[df['Model'] != 'Ensemble'].pivot_table(
        index='Model', columns='Period', values='MAE', aggfunc='mean'
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), max(4, len(pivot.index) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', linewidths=0.5, ax=ax)
    ax.set_title("Mean MAE — Model × Period (lower = better)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "plot_MAE_heatmap.png"), dpi=150)
    plt.close()

def plot_horizon_decay(df, out_folder):
    """Line plot: R² vs Horizon for each model."""
    if df['Horizon'].nunique() < 2:
        return

    for period in df['Period'].unique():
        sub = df[df['Period'] == period]
        avg = sub.groupby(['Model', 'Horizon'])['R2'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        for model in avg['Model'].unique():
            m_data = avg[avg['Model'] == model].sort_values('Horizon')
            if model == 'ARIMA':
                marker, ls, lw = 'D', ':', 2.5
            elif model == 'Ensemble':
                marker, ls, lw = 's', '--', 2
            else:
                marker, ls, lw = 'o', '-', 2
            ax.plot(m_data['Horizon'], m_data['R2'], marker=marker, ls=ls, label=model, linewidth=lw)

        ax.set_xlabel("Forecast Horizon (days)")
        ax.set_ylabel("R²")
        ax.set_title(f"R² vs Horizon — {period}")
        ax.legend(loc='best', fontsize=9)
        ax.axhline(0, color='black', linewidth=0.5, ls='--')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = period.replace(' ', '_')
        plt.savefig(os.path.join(out_folder, f"plot_horizon_decay_{safe}.png"), dpi=150)
        plt.close()

def plot_window_comparison(df, out_folder):
    """Grouped bar: R² per Window, grouped by Model."""
    if df['Window'].nunique() < 2:
        return

    for period in df['Period'].unique():
        for h in df['Horizon'].unique():
            sub = df[(df['Period'] == period) & (df['Horizon'] == h)]
            if sub.empty or sub['Model'].nunique() < 2:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Window', y='R2', hue='Model', data=sub, ax=ax)
            ax.set_title(f"R² by Window | {period} | H={h}")
            ax.axhline(0, color='black', linewidth=0.5)
            ax.legend(loc='best', fontsize=9)
            plt.tight_layout()
            safe = period.replace(' ', '_')
            plt.savefig(os.path.join(out_folder, f"plot_window_{safe}_H{h}.png"), dpi=150)
            plt.close()

def plot_model_radar(df, out_folder):
    """Radar/spider chart: multi-metric comparison across models."""
    models = [m for m in df['Model'].unique() if m != 'Ensemble']
    if len(models) < 2:
        return

    # Compute normalized scores for each model
    metrics = {}
    for model in models:
        sub = df[df['Model'] == model]
        metrics[model] = {
            'Mean R²': sub['R2'].mean(),
            '1 - MAE': 1 - sub['MAE'].mean(),  # invert so higher = better
            'R² Stability\n(1-Std)': 1 - sub['R2'].std(),
            'Best R²': sub['R2'].max(),
            'Worst R²': max(0, sub['R2'].min()),  # clip negative
        }

    df_metrics = pd.DataFrame(metrics).T
    # Normalize each column to [0, 1]
    for col in df_metrics.columns:
        cmin, cmax = df_metrics[col].min(), df_metrics[col].max()
        if cmax > cmin:
            df_metrics[col] = (df_metrics[col] - cmin) / (cmax - cmin)
        else:
            df_metrics[col] = 1.0

    categories = list(df_metrics.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        values = df_metrics.loc[model].values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison Radar", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "plot_model_radar.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_period_bar_comparison(df, out_folder):
    """Side-by-side bar chart: R² per model, grouped by period."""
    models = [m for m in df['Model'].unique() if m != 'Ensemble']
    periods = df['Period'].unique()
    if len(periods) < 1 or len(models) < 1:
        return

    avg = df[df['Model'] != 'Ensemble'].groupby(['Model', 'Period'])['R2'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(max(10, len(periods) * 2.5), 6))
    sns.barplot(x='Period', y='R2', hue='Model', data=avg, ax=ax)
    ax.set_title("Mean R² — Model × Period")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "plot_period_bar_R2.png"), dpi=150)
    plt.close()


def plot_vs_arima_bar(df, out_folder):
    """Bar chart: R² improvement of each DL model over ARIMA baseline."""
    if 'ARIMA' not in df['Model'].values:
        return

    other_models = [m for m in df['Model'].unique() if m not in ('ARIMA', 'Ensemble')]
    if not other_models:
        return

    arima_avg = df[df['Model'] == 'ARIMA'].groupby('Period')['R2'].mean()
    rows = []
    for model in other_models:
        model_avg = df[df['Model'] == model].groupby('Period')['R2'].mean()
        for period in model_avg.index:
            if period in arima_avg.index:
                rows.append({
                    'Model': model, 'Period': period,
                    'Delta_R2': model_avg[period] - arima_avg[period]
                })

    if not rows:
        return

    df_delta = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(10, df_delta['Period'].nunique() * 2.5), 6))
    sns.barplot(x='Period', y='Delta_R2', hue='Model', data=df_delta, ax=ax)
    ax.set_title("R² Improvement over ARIMA Baseline")
    ax.set_ylabel("Delta R² (positive = beats ARIMA)")
    ax.axhline(0, color='red', linewidth=1, ls='--', label='ARIMA baseline')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "plot_vs_arima_delta_R2.png"), dpi=150)
    plt.close()

# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("  RESULT ANALYZER")
    print("=" * 50)

    all_runs = discover_folders()
    if not all_runs:
        print("[ERROR] No result folders found. Run SpecialPeriod.py first.")
        sys.exit(1)

    selected = select_runs(all_runs)
    if not selected:
        print("[ERROR] No runs selected.")
        sys.exit(1)

    # Load all selected runs
    dfs = []
    for run in selected:
        try:
            dfs.append(load_run(run))
            print(f"  Loaded: {run['type']} Seed={run['seed']} ({len(dfs[-1])} rows)")
        except Exception as e:
            print(f"  [WARN] Failed to load {run['folder']}: {e}")

    if not dfs:
        print("[ERROR] No data loaded.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n[DATA] Combined: {len(df)} rows | {df['Model'].nunique()} models | "
          f"{df['Period'].nunique()} periods | {df['Seed'].nunique()} seeds")

    # Output folder
    seeds_str = "_".join(sorted(set(r['seed'] for r in selected)))
    OUT_FOLDER = os.path.join(BASE_DIR, f"Analysis_{seeds_str}")
    os.makedirs(OUT_FOLDER, exist_ok=True)

    # Run all analyses
    analyze_overview(df, OUT_FOLDER)
    df_rank = analyze_model_ranking(df, OUT_FOLDER)
    analyze_horizon_sensitivity(df, OUT_FOLDER)
    analyze_window_sensitivity(df, OUT_FOLDER)
    flags = analyze_special_flags(df, OUT_FOLDER)
    analyze_vs_arima_baseline(df, OUT_FOLDER)
    analyze_cross_period_robustness(df, OUT_FOLDER)
    analyze_cross_seed(df, OUT_FOLDER)

    # Generate plots
    print_divider("GENERATING PLOTS")
    plot_r2_heatmap(df, OUT_FOLDER)
    plot_mae_heatmap(df, OUT_FOLDER)
    plot_horizon_decay(df, OUT_FOLDER)
    plot_window_comparison(df, OUT_FOLDER)
    plot_model_radar(df, OUT_FOLDER)
    plot_period_bar_comparison(df, OUT_FOLDER)
    plot_vs_arima_bar(df, OUT_FOLDER)
    print("  All plots saved.")

    # Save combined data
    df.to_csv(os.path.join(OUT_FOLDER, "combined_data.csv"), index=False)

    # Final summary
    print_divider("ANALYSIS COMPLETE")
    print(f"  Output folder: {OUT_FOLDER}")
    print(f"  Files generated:")
    for f in sorted(os.listdir(OUT_FOLDER)):
        size = os.path.getsize(os.path.join(OUT_FOLDER, f))
        print(f"    {f} ({size:,} bytes)")
