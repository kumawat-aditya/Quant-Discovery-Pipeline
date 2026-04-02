# app.py (V11 - Robust Visualization Engine)

"""
The Strategy Post-Mortem Dashboard (The Analyser)

This script is the final, human-led analysis layer of the entire strategy
discovery pipeline. It launches a powerful, interactive Streamlit web application
designed for the deep interrogation of surviving strategies.

Its purpose is to provide a "post-mortem" view of each strategy, allowing the
user to go beyond simple metrics and understand the *why* behind a strategy's
performance. It helps answer critical questions like:
- Was the strategy's edge on its home market real or a fluke?
- Under what specific market conditions (regimes) does it thrive or fail?
- Did the strategy get lucky by aligning with a strong market trend?

This version includes robust visualization logic to gracefully handle edge cases
like infinite Profit Factors (from 100% win rates) and zero-drawdown performance,
ensuring all charts are stable and readable.
"""

import streamlit as st
import pandas as pd
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import numpy as np

# --- Page Configuration & Constants ---
st.set_page_config(page_title="Strategy Post-Mortem", page_icon="🕵️", layout="wide")
# A cap for Sharpe Ratio values to prevent extreme outliers from skewing charts.
SHARPE_CAP = 10.0
# A display cap for Profit Factor for consistent visualization of 'infinite' PF.
PF_CAP = 100.0

# --- Path Configuration ---
CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIAMOND_RESULTS_DIR = os.path.join(CORE_DIR, 'diamond_data', 'backtesting_results')
ZIRCON_RESULTS_DIR = os.path.join(CORE_DIR, 'zircon_data', 'results')
BLACKLIST_DIR = os.path.join(CORE_DIR, 'platinum_data', 'blacklists')
PREPARED_DATA_DIR = os.path.join(CORE_DIR, 'diamond_data', 'prepared_data')
TRADE_LOGS_DIR = os.path.join(CORE_DIR, 'zircon_data', 'trade_logs')

# --- Data Loading & Helper Functions ---

@st.cache_data(ttl=600)
def get_available_reports():
    """
    Scans the Zircon results directory to find all available summary reports.

    This function uses a regular expression to identify files that match the
    `summary_report_*.csv` pattern, extracting the market name from the filename.
    The results are cached to avoid repeated disk I/O on every interaction.

    Returns:
        list: A sorted list of report names (e.g., ['XAUUSD15', 'EURUSD60'])
              available for analysis.
    """
    if not os.path.exists(ZIRCON_RESULTS_DIR): return []
    pattern = re.compile(r"summary_report_(.+)\.csv")
    files = os.listdir(ZIRCON_RESULTS_DIR)
    return sorted([pattern.match(f).group(1) for f in files if pattern.match(f)])

@st.cache_data(ttl=600)
def load_and_merge_data(report_name):
    """
    Loads and merges all relevant data sources for a selected report.

    This is the primary data loading function for the dashboard. It fetches the
    Zircon summary report (cross-market performance), the Zircon detailed report
    (per-market performance), and the Diamond mastery report (origin-market
    performance). It then merges them into a single comprehensive DataFrame.
    Crucially, it cleans the data by capping extreme values for Profit Factor and
    Sharpe Ratio to ensure visualizations are stable and readable.

    Args:
        report_name (str): The name of the report to load, corresponding to a
                           specific instrument and timeframe (e.g., 'XAUUSD15').

    Returns:
        tuple: A tuple of three DataFrames:
               - master_view_df: The main merged and cleaned DataFrame for the dashboard.
               - zircon_detailed_df: The raw detailed report from the Zircon layer.
               - mastery_df: The raw report from the Diamond layer.
    """
    if not report_name: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    zircon_summary_path = os.path.join(ZIRCON_RESULTS_DIR, f"summary_report_{report_name}.csv")
    zircon_detailed_path = os.path.join(ZIRCON_RESULTS_DIR, f"detailed_report_{report_name}.csv")
    diamond_report_path = os.path.join(DIAMOND_RESULTS_DIR, f"diamond_report_{report_name}.csv")

    if not os.path.exists(zircon_summary_path): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    zircon_summary_df = pd.read_csv(zircon_summary_path)
    zircon_detailed_df = pd.read_csv(zircon_detailed_path)
    mastery_df = pd.read_csv(diamond_report_path) if os.path.exists(diamond_report_path) else pd.DataFrame()

    zircon_renamed = zircon_summary_df.rename(columns={'avg_profit_factor': 'validation_avg_pf', 'avg_sharpe_ratio': 'validation_avg_sharpe', 'avg_max_drawdown_pct': 'validation_avg_dd_pct', 'total_trades': 'validation_total_trades', 'validation_markets_passed': 'validation_passed'})
    
    if not mastery_df.empty:
        mastery_renamed = mastery_df.rename(columns={'profit_factor': 'mastery_pf', 'sharpe_ratio': 'mastery_sharpe', 'max_drawdown_pct': 'mastery_dd_pct', 'total_trades': 'mastery_total_trades'})
        master_view_df = pd.merge(zircon_renamed, mastery_renamed[['strategy_id', 'mastery_pf', 'mastery_sharpe', 'mastery_dd_pct', 'mastery_total_trades']], on='strategy_id', how='left')
    else:
        master_view_df = zircon_renamed.copy()
        for col in ['mastery_pf', 'mastery_sharpe', 'mastery_dd_pct', 'mastery_total_trades']: master_view_df[col] = np.nan
    
    # VISUALIZATION FIX: Clean and cap values for better charts.
    for col in ['mastery_pf', 'validation_avg_pf']:
        if col in master_view_df.columns: master_view_df[col] = master_view_df[col].replace([np.inf, -np.inf], PF_CAP)
    for col in ['mastery_sharpe', 'validation_avg_sharpe']:
        if col in master_view_df.columns: master_view_df[col] = master_view_df[col].replace([np.inf, -np.inf], SHARPE_CAP).clip(upper=SHARPE_CAP)
        
    return master_view_df, zircon_detailed_df, mastery_df

def write_to_blacklist(strategy_key, report_name):
    """
    Appends a strategy's parent blueprint key to the corresponding blacklist file.

    This function provides the feedback mechanism from the dashboard back to the
    discovery pipeline. It takes a strategy's unique 'key', loads the existing
    blacklist for that instrument, adds the new key, removes duplicates, and
    saves the file. This prevents the flawed parent blueprint from being analyzed
    in future runs of the Platinum layer.

    Args:
        strategy_key (str): The unique hash key of the strategy's parent blueprint.
        report_name (str): The name of the origin market report (e.g., 'XAUUSD15'),
                           used to identify the correct blacklist file.
    """
    blacklist_path = os.path.join(BLACKLIST_DIR, f"{report_name}.csv")
    key_to_add = pd.DataFrame([{'key': strategy_key}])
    try:
        blacklist_df = pd.read_csv(blacklist_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        blacklist_df = pd.DataFrame(columns=['key'])
    updated_blacklist = pd.concat([blacklist_df, key_to_add], ignore_index=True).drop_duplicates()
    updated_blacklist.to_csv(blacklist_path, index=False)
    st.toast(f"Blueprint key {strategy_key[:10]}... blacklisted in {report_name}.csv!")
    st.cache_data.clear()

@st.cache_data(ttl=3600)
def load_market_internals(markets):
    """
    Loads prepared Silver data to calculate high-level market characteristics.

    For each market provided, this function reads the pre-calculated Silver
    data and computes metrics like the percentage of time spent in a trend,
    the percentage of time in high volatility, and the overall price change.
    This provides crucial context to determine if a strategy's success was
    simply due to a strong market tailwind.

    Args:
        markets (list): A list of market names (e.g., ['EURUSD15.csv', 'GBPUSD15.csv']).

    Returns:
        pd.DataFrame: A DataFrame summarizing the internal characteristics of each market.
    """
    internals = []
    for market_name in markets:
        silver_path = os.path.join(PREPARED_DATA_DIR, f"{market_name.replace('.csv','')}_silver.parquet")
        if os.path.exists(silver_path):
            df = pd.read_parquet(silver_path, columns=['trend_regime', 'vol_regime', 'BB_width', 'close'])
            internals.append({
                'market': market_name, 
                '% Time in Trend': df['trend_regime'].value_counts(normalize=True).get('trend', 0) * 100,
                '% Time High Vol': df['vol_regime'].value_counts(normalize=True).get('high_vol', 0) * 100,
                'Avg BB Width': df['BB_width'].mean(),
                'Overall Price Change %': (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            })
    return pd.DataFrame(internals)

@st.cache_data(ttl=3600)
def load_full_silver_data(market_name):
    """
    Loads the time and close price from a market's prepared Silver data.

    This is a lightweight helper function designed to quickly fetch the necessary
    data for plotting a market's price history. The data is cached for performance.

    Args:
        market_name (str): The name of the market to load (e.g., 'XAUUSD15.csv').

    Returns:
        pd.DataFrame or None: A DataFrame with 'time' and 'close' columns, or None
                              if the data file cannot be found.
    """
    silver_path = os.path.join(PREPARED_DATA_DIR, f"{market_name.replace('.csv','')}_silver.parquet")
    return pd.read_parquet(silver_path, columns=['time', 'close']) if os.path.exists(silver_path) else None

@st.cache_data(ttl=600)
def load_trade_log(strategy_id, market_name):
    """
    Loads the detailed, trade-by-trade log for a specific strategy on a specific market.

    This function locates and reads the CSV file containing every individual trade
    executed during a strategy's backtest on a given market. This granular data
    powers the most detailed visualizations, such as plotting trade entries on
    the price chart.

    Args:
        strategy_id (str): The unique identifier of the strategy.
        market_name (str): The market on which the strategy was tested (e.g., 'XAUUSD15.csv').

    Returns:
        pd.DataFrame or None: A DataFrame containing the detailed trade log, or None
                              if the log file does not exist.
    """
    log_path = os.path.join(TRADE_LOGS_DIR, strategy_id, f"{market_name.replace('.csv','')}.csv")
    return pd.read_csv(log_path, parse_dates=['entry_time']) if os.path.exists(log_path) else None

def parse_dict_col(data_string):
    """
    Safely parses a string representation of a dictionary into a Python dictionary.

    This utility function uses `ast.literal_eval` to securely convert a string
    (e.g., "{'London': 75.0, 'Asian': 25.0}") into a dictionary object. It includes
    error handling to return an empty dictionary if parsing fails, preventing crashes.

    Args:
        data_string (str): The string to be parsed.

    Returns:
        dict: The parsed dictionary, or an empty dictionary if parsing fails.
    """
    try: return ast.literal_eval(str(data_string))
    except: return {}

# --- UI START ---

st.title("🕵️ Strategy Post-Mortem Dashboard")

# --- Sidebar Controls ---
st.sidebar.title("Controls")
available_reports = get_available_reports()
if not available_reports:
    st.error(f"No Zircon summary reports found in `{ZIRCON_RESULTS_DIR}`"); st.stop()

selected_report = st.sidebar.selectbox("Select Report to Analyze", options=available_reports, key="report_selector")
master_df, detailed_df, mastery_df = load_and_merge_data(selected_report)
if master_df.empty:
    st.warning(f"Could not load data for **{selected_report}**."); st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filter Strategies")
if st.sidebar.button("Reset All Filters"):
    st.session_state.m_pf, st.session_state.m_sh, st.session_state.v_pf = 1.5, 1.0, 1.0
    st.session_state.v_pass = '0/0'; st.rerun()

st.sidebar.markdown("**Mastery Filters (Origin Market)**")
mastery_pf_min = st.sidebar.slider("Min Mastery PF", 0.0, 10.0, 1.5, 0.1, key='m_pf')
mastery_sharpe_min = st.sidebar.slider("Min Mastery Sharpe", 0.0, SHARPE_CAP, 1.0, 0.1, key='m_sh')
st.sidebar.markdown("---")
st.sidebar.markdown("**Validation Filters (Other Markets)**")
validation_pf_min = st.sidebar.slider("Min Validation Avg PF", 0.0, 5.0, 1.0, 0.1, key='v_pf')
pass_options = sorted(master_df['validation_passed'].dropna().unique(), key=lambda x: (int(x.split('/')[1]), int(x.split('/')[0])), reverse=True)
min_markets_passed_str = st.sidebar.selectbox("Min Validation Markets Passed", options=pass_options, key='v_pass')

min_passed_count = int(min_markets_passed_str.split('/')[0])
filtered_df = master_df.copy()
for col in ['mastery_pf', 'mastery_sharpe', 'validation_avg_pf', 'validation_markets_passed_count']:
    filtered_df[col] = filtered_df[col].fillna(0)
filtered_df = filtered_df[
    (filtered_df['mastery_pf'] >= mastery_pf_min) &
    (filtered_df['mastery_sharpe'] >= mastery_sharpe_min) &
    (filtered_df['validation_avg_pf'] >= validation_pf_min) &
    (filtered_df['validation_markets_passed_count'] >= min_passed_count)
]

# --- Main Content with Tabs ---
tab1, tab2 = st.tabs(["🏆 Strategy Dashboard", "🕵️ Post-Mortem Analysis"])

with tab1:
    st.header(f"Strategy Dashboard for `{selected_report}`")
    st.info("Use the sidebar filters to discover robust strategies. The table below shows the survivors.")
    st.metric("Robust Strategies Found", f"{len(filtered_df)} / {len(master_df)}")
    
    display_cols = ['strategy_id', 'mastery_pf', 'mastery_sharpe', 'validation_avg_pf', 'validation_passed', 'market_rule']
    st.dataframe(filtered_df[display_cols].rename(columns=lambda c: c.replace('_', ' ').title()))

with tab2:
    st.header("Post-Mortem Analysis")
    if filtered_df.empty:
        st.warning("No strategies match filters. Please relax filters in the sidebar.")
    else:
        strategy_id = st.selectbox("Select a Strategy for Interrogation", filtered_df['strategy_id'].unique())
        if strategy_id:
            strategy_data = filtered_df.loc[filtered_df['strategy_id'] == strategy_id].iloc[0]
            validation_data = detailed_df[detailed_df['strategy_id'] == strategy_id]
            mastery_data_row = mastery_df[mastery_df['strategy_id'] == strategy_id]
            mastery_data = mastery_data_row.iloc[0] if not mastery_data_row.empty else None

            # --- Section 1: Strategy Profile ---
            st.subheader("1. Strategy Profile")
            sl_def = f"{strategy_data['sl_def']:.3f}%" if isinstance(strategy_data['sl_def'], float) else f"{strategy_data['sl_def']} (Bin {strategy_data['sl_bin'] or 'N/A'})".replace('.0', '')
            tp_def = f"{strategy_data['tp_def']:.3f}%" if isinstance(strategy_data['tp_def'], float) else f"{strategy_data['tp_def']} (Bin {strategy_data['tp_bin'] or 'N/A'})".replace('.0', '')
            st.metric("Trade Type", strategy_data.get('trade_type', 'N/A').upper())
            st.text_input("Stop-Loss Logic", sl_def, disabled=True)
            st.text_input("Take-Profit Logic", tp_def, disabled=True)
            st.code(strategy_data['market_rule'], language='sql')
            st.sidebar.markdown("---")
            st.sidebar.header("Manual Actions")
            if st.sidebar.button("🚫 Manually Blacklist this Blueprint", type="primary", key=f"bl_{strategy_id}"):
                write_to_blacklist(strategy_data['key'], selected_report)
                st.rerun()

            # --- Section 2: The Verdict ---
            st.subheader("2. The Verdict: Was the Edge Real?")
            v_col1, v_col2 = st.columns(2)
            if mastery_data is not None:
                v_col1.metric("Mastery Profit Factor", f"{mastery_data.get('profit_factor', 0):.2f}".replace(f"{PF_CAP:.2f}", "💯 Inf"))
                v_col1.metric("Mastery Sharpe Ratio", f"{mastery_data.get('sharpe_ratio', 0):.2f}")
            v_col2.metric("Validation Avg Profit Factor", f"{strategy_data['validation_avg_pf']:.2f}".replace(f"{PF_CAP:.2f}", "💯 Inf"))
            v_col2.metric("Markets Passed", strategy_data['validation_passed'])

            # --- Section 3: Performance Breakdown ---
            st.subheader("3. Evidence: Performance Breakdown")
            metric_map = {'Profit Factor': 'profit_factor', 'Sharpe Ratio': 'sharpe_ratio', 'Max Drawdown %': 'max_drawdown_pct'}
            selected_metric = metric_map[st.radio("Select Metric to Compare:", metric_map.keys(), horizontal=True, key="metric_radio")]
            
            full_perf_data = pd.concat([pd.DataFrame([mastery_data]), validation_data], ignore_index=True) if mastery_data is not None else validation_data
            
            # VISUALIZATION FIX: Handle inf, NaN, and 0 for stable plotting.
            full_perf_data[selected_metric] = full_perf_data[selected_metric].replace([np.inf, -np.inf], PF_CAP if 'profit' in selected_metric else SHARPE_CAP).fillna(0)
            
            y_range = None
            if selected_metric == 'max_drawdown_pct':
                y_range = [0, max(1, full_perf_data[selected_metric].max() * 1.1)] # Ensure y-axis starts at 0

            fig_compare = px.bar(full_perf_data, x='market', y=selected_metric, color=selected_metric, title=f"<b>{selected_metric.replace('_',' ').title()} Across Markets</b>", color_continuous_scale='RdYlGn' if 'drawdown' not in selected_metric else 'RdYlGn_r', range_y=y_range)
            fig_compare.add_hline(y=1.0 if selected_metric == 'profit_factor' else 0.0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_compare, use_container_width=True)

            # --- Section 4: Deep Dive into Market Regimes ---
            if 'session_pct' in full_perf_data.columns:
                st.subheader("4. Deep Dive: Regime Performance")
                regime_type = st.radio("Select Regime to Analyze", ['session', 'trend_regime', 'vol_regime'], horizontal=True, format_func=lambda x: x.replace('_', ' ').title())
                
                regime_data = []
                for _, row in full_perf_data.iterrows():
                    # VISUALIZATION FIX: Handle NaN/inf for color metric.
                    pf_for_color = row['profit_factor'] if pd.notna(row['profit_factor']) else 1.0
                    pf_for_color = min(pf_for_color, 5.0) # Cap PF for color scale
                    
                    for regime, pct in parse_dict_col(row[f'{regime_type}_pct']).items():
                        regime_data.append({'market': row['market'], 'Regime': regime, 'Trade Percentage': pct, 'Profit Factor': pf_for_color})
                regime_df = pd.DataFrame(regime_data)

                fig_regime = px.bar(regime_df, x='Regime', y='Trade Percentage', color='Profit Factor', color_continuous_scale='RdYlGn', range_color=[0, 2], facet_col='market', labels={'Trade Percentage': '% of Trades'}, title=f"<b>Performance by {regime_type.replace('_', ' ').title()}</b> (Color is Overall Market PF)")
                st.plotly_chart(fig_regime, use_container_width=True)

            # --- Section 5: Market Internals vs. Performance ---
            st.subheader("5. Market Internals: Did the Strategy Get Lucky?")
            all_markets = full_perf_data['market'].unique()
            market_internals_df = load_market_internals(all_markets)
            if not market_internals_df.empty:
                internals_vs_perf = pd.merge(market_internals_df, full_perf_data[['market', 'profit_factor']], on='market', how='left').fillna(0)
                internal_col1, internal_col2 = st.columns(2)
                with internal_col1:
                    fig_internal1 = px.scatter(internals_vs_perf, x='% Time in Trend', y='profit_factor', color='market', title="Performance vs. Market's Tendency to Trend", hover_name='market')
                    st.plotly_chart(fig_internal1, use_container_width=True)
                with internal_col2:
                    fig_internal2 = px.scatter(internals_vs_perf, x='Overall Price Change %', y='profit_factor', color='market', title="Performance vs. Market's Overall Direction", hover_name='market')
                    st.plotly_chart(fig_internal2, use_container_width=True)
            else:
                st.warning(f"Could not load Silver data to analyze market internals. Check path: `{PREPARED_DATA_DIR}`")
            
            # --- Section 6: Performance in Context of Market Trend ---
            st.subheader("6. Performance in Context: Market Trend")
            st.info("Visualize individual trades (Green=Win, Red=Loss) overlaid on the market's price history to see how the strategy performs in different macro trends.")

            all_markets_for_strat = sorted(list(set(full_perf_data['market'].unique())))

            for market_name in all_markets_for_strat:
                st.markdown(f"#### Trend Context for `{market_name}`")
                
                silver_df = load_full_silver_data(market_name)
                trade_log_df = load_trade_log(strategy_id, market_name)

                if silver_df is None:
                    st.warning(f"Could not load Silver data for {market_name}."); continue
                
                fig = go.Figure()
                # Plot the base close price line
                fig.add_trace(go.Scatter(x=silver_df['time'], y=silver_df['close'], mode='lines', name='Close Price', line=dict(color='rgba(128, 128, 128, 0.5)')))

                if trade_log_df is not None:
                    wins = trade_log_df[trade_log_df['pnl'] > 0]
                    losses = trade_log_df[trade_log_df['pnl'] <= 0]
                    
                    # Plot winning trades as green upward-pointing triangles
                    fig.add_trace(go.Scatter(
                        x=wins['entry_time'], y=wins['entry_price'],
                        mode='markers', name='Winning Trades',
                        marker=dict(color='limegreen', size=8, symbol='triangle-up', line=dict(width=1, color='DarkSlateGrey'))
                    ))
                    # Plot losing trades as red downward-pointing triangles
                    fig.add_trace(go.Scatter(
                        x=losses['entry_time'], y=losses['entry_price'],
                        mode='markers', name='Losing Trades',
                        marker=dict(color='red', size=8, symbol='triangle-down', line=dict(width=1, color='DarkSlateGrey'))
                    ))
                else:
                    st.info(f"Trade log not found for this market. Run the Zircon Validator to generate it.")
                
                fig.update_layout(title=f"Trade Entries vs. Market Trend for {market_name}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)