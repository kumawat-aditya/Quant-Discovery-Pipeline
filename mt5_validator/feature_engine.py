# File: mt5_validator/feature_engine.py

"""
Replicates the Silver and Gold layer feature generation for live data.

This module is critical for ensuring consistency between research and live
environments. It takes a raw OHLC DataFrame from the DataEngine and applies
the exact same feature calculation, transformation, and scaling logic as
the original `silver_data_generator.py` and `gold_data_generator.py` scripts.
"""

import pandas as pd
import numpy as np
import logging
import re
import ta
import talib
import numba
from typing import Tuple, List

# We must import the RESEARCH config file to access the indicator parameters
# to ensure they are identical.
import sys
import os

# Temporarily add the project root to the path to import the research config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
try:
    import config as research_config
finally:
    sys.path.pop(0) # Clean up path after import

# Set up a logger for this module
logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    A class to orchestrate the full feature generation pipeline on live data.
    """

    def __init__(self):
        """Initializes the FeatureEngine."""
        # This engine is stateless, but a class structure is used for clarity.
        logger.info("FeatureEngine initialized. Using indicator parameters from research config.")
        
        # Pre-compile the regex pattern for relational feature transformation
        self.abs_price_patterns = re.compile(
            r'^(open|high|low|close)$|^(SMA|EMA)_\d+$|^BB_(upper|lower)_\d+$|'
            r'^(support|resistance)$|^ATR_level_.+_\d+$'
        )
        
        # Pre-fetch the list of candlestick pattern function names from TA-Lib
        self.candle_pattern_names = talib.get_function_groups().get("Pattern Recognition", [])

    # --- STAGE 1: SILVER FEATURE GENERATION ---

    @staticmethod
    @numba.njit
    def _calculate_s_r_numba(lows: np.ndarray, highs: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Identifies fractal support and resistance points using Numba. (Copied from Silver)"""
        n = len(lows)
        support = np.full(n, np.nan, dtype=np.float32)
        resistance = np.full(n, np.nan, dtype=np.float32)
        for i in range(window, n - window):
            ws = slice(i - window, i + window + 1)
            if lows[i] == np.min(lows[ws]):
                support[i] = lows[i]
            if highs[i] == np.max(highs[ws]):
                resistance[i] = highs[i]
        return support, resistance

    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates and adds forward-filled S/R levels. (Copied from Silver)"""
        lows = df["low"].values.astype(np.float32)
        highs = df["high"].values.astype(np.float32)
        support_pts, resistance_pts = self._calculate_s_r_numba(lows, highs, research_config.PIVOT_WINDOW)
        df["support"] = pd.Series(support_pts, index=df.index).ffill()
        df["resistance"] = pd.Series(resistance_pts, index=df.index).ffill()
        return df

    @staticmethod
    def _map_market_sessions(hour_series: pd.Series) -> pd.Series:
        """Maps UTC hours to Forex market sessions. (Copied from Silver)"""
        bins = [-1, 0, 8, 9, 13, 17, 22, 23]
        labels = ['Tokyo', 'Tokyo_London_Overlap', 'London', 'London_NY_Overlap', 'New_York', 'Sydney', 'Sydney']
        return pd.cut(hour_series, bins=bins, labels=labels, ordered=False, right=True)

    @staticmethod
    def _add_standard_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates a batch of standard technical indicators. (Copied from Silver)"""
        indicator_df = pd.DataFrame(index=df.index)
        # --- This logic is a 1:1 copy from silver_data_generator.py ---
        for p in research_config.SMA_PERIODS:
            indicator_df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
        for p in research_config.EMA_PERIODS:
            indicator_df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()
        for p in research_config.BBANDS_PERIODS:
            bb = ta.volatility.BollingerBands(df["close"], p, research_config.BBANDS_STD_DEV)
            indicator_df[f"BB_upper_{p}"] = bb.bollinger_hband()
            indicator_df[f"BB_lower_{p}"] = bb.bollinger_lband()
        for p in research_config.RSI_PERIODS:
            indicator_df[f"RSI_{p}"] = ta.momentum.RSIIndicator(df["close"], p).rsi()
        macd_key = f"MACD_hist_{research_config.MACD_FAST}_{research_config.MACD_SLOW}_{research_config.MACD_SIGNAL}"
        indicator_df[macd_key] = ta.trend.MACD(df["close"], research_config.MACD_SLOW, research_config.MACD_FAST, research_config.MACD_SIGNAL).macd_diff()
        for p in research_config.ATR_PERIODS:
            indicator_df[f"ATR_{p}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], p).average_true_range()
        for p in research_config.ADX_PERIODS:
            indicator_df[f"ADX_{p}"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], p).adx()

        indicator_df["MOM_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()
        indicator_df["CCI_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
        
        for p in research_config.ATR_PERIODS:
            atr_series = indicator_df[f"ATR_{p}"]
            indicator_df[f"ATR_level_up_1x_{p}"] = df["close"] + atr_series
            indicator_df[f"ATR_level_down_1x_{p}"] = df["close"] - atr_series
        return indicator_df

    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates TA-Lib candlestick patterns. (Copied from Silver)"""
        patterns_df = pd.DataFrame(index=df.index)
        for p in self.candle_pattern_names:
            try:
                patterns_df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"])
            except Exception:
                patterns_df[p] = 0 # Handle potential errors gracefully
        return patterns_df

    def _add_time_and_pa_features(self, df: pd.DataFrame, indicator_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates time-based and price-action features. (Copied from Silver)"""
        pa_df = pd.DataFrame(index=df.index)
        pa_df['session'] = self._map_market_sessions(df['time'].dt.hour)
        pa_df['hour'], pa_df['weekday'] = df['time'].dt.hour, df['time'].dt.weekday
        is_bullish = (df['close'] > df['open']).astype(int)
        body_size = np.abs(df['close'] - df['open'])
        for n in research_config.PAST_LOOKBACKS:
            pa_df[f'bullish_ratio_last_{n}'] = is_bullish.rolling(n, min_periods=1).mean()
            pa_df[f'avg_body_last_{n}'] = body_size.rolling(n, min_periods=1).mean()
            pa_df[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n, min_periods=1).mean()
        for p in research_config.ADX_PERIODS:
            pa_df[f'trend_regime_{p}'] = np.where(indicator_df[f'ADX_{p}'] > 25, 'trend', 'range')
        for p in research_config.ATR_PERIODS:
            atr_rolling_mean = indicator_df[f'ATR_{p}'].rolling(50, min_periods=1).mean()
            pa_df[f'vol_regime_{p}'] = np.where(indicator_df[f'ATR_{p}'] > atr_rolling_mean, 'high_vol', 'low_vol')
        return pa_df

    def calculate_silver_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Orchestrates the calculation of all Silver-level features."""
        df = raw_df.copy()
        indicator_df = self._add_standard_indicators(df)
        patterns_df = self._add_candlestick_patterns(df)
        df = self._add_support_resistance(df)
        pa_df = self._add_time_and_pa_features(df, indicator_df)
        return pd.concat([df, indicator_df, patterns_df, pa_df], axis=1)

    # --- STAGE 2: GOLD FEATURE GENERATION ---

    def _transform_relational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts absolute price columns to be relative. (Copied from Gold)"""
        abs_price_cols = [col for col in df.columns if self.abs_price_patterns.match(col)]
        close_series = df['close']
        for col in abs_price_cols:
            if col != 'close':
                df[f'{col}_dist_norm'] = (df[col] - close_series) / close_series.replace(0, np.nan)
        df.drop(columns=abs_price_cols, inplace=True, errors='ignore')
        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """One-hot encodes categorical columns. (Copied from Gold)"""
        categorical_cols = [col for col in df.columns if col.startswith(('session', 'trend_regime', 'vol_regime'))]
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=float)
        return df, categorical_cols

    def _compress_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bins candlestick scores into a 5-point scale. (Copied from Gold)"""
        candle_cols = [col for col in df.columns if col.startswith("CDL")]
        compress_func = np.vectorize(lambda v: 1.0 if v >= 100 else (0.5 if v > 0 else (-1.0 if v <= -100 else (-0.5 if v < 0 else 0.0))))
        for col in candle_cols:
            df[col] = compress_func(df[col].fillna(0))
        return df

    def _scale_numeric_features(self, df: pd.DataFrame, original_cat_cols: List[str]) -> pd.DataFrame:
        """Standardizes features using a rolling window. (Copied from Gold)"""
        candle_cols = {col for col in df.columns if col.startswith("CDL")}
        one_hot_cols = {col for col in df.columns if any(cat in col for cat in original_cat_cols)}
        non_scalable_cols = candle_cols | one_hot_cols | {'time'}
        cols_to_scale = [col for col in df.columns if col not in non_scalable_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        window_size = research_config.GOLD_SCALER_ROLLING_WINDOW
        for col in cols_to_scale:
            df[col] = df[col].fillna(0)
            rolling_mean = df[col].rolling(window=window_size, min_periods=2).mean()
            rolling_std = df[col].rolling(window=window_size, min_periods=2).std()
            df[col] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
            df[col] = df[col].bfill().fillna(0)
        return df

    # --- PUBLIC ORCHESTRATOR METHOD ---
    
    def generate_gold_features(self, raw_ohlc_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        The main public method to run the full feature generation pipeline.

        Args:
            raw_ohlc_df (pd.DataFrame): The raw OHLC DataFrame from the DataEngine.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                1. The fully processed Gold-level DataFrame for signal evaluation.
                2. The intermediate Silver-level DataFrame, needed for calculating
                   absolute SL/TP prices.
        """
        logger.debug(f"Starting feature generation for {len(raw_ohlc_df)} bars.")
        
        # Stage 1: Generate Silver features
        silver_df = self.calculate_silver_features(raw_ohlc_df)
        
        # Stage 2: Transform Silver to Gold
        gold_df = silver_df.copy()
        gold_df = self._transform_relational_features(gold_df)
        gold_df, original_cats = self._encode_categorical_features(gold_df)
        gold_df = self._compress_candlestick_patterns(gold_df)
        gold_df = self._scale_numeric_features(gold_df, original_cats)
        
        logger.debug("Feature generation complete.")
        return gold_df, silver_df