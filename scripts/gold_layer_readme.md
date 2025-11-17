# Gold Layer: The Machine Learning Preprocessor

The script `gold_data_generator.py` represents the crucial final stage of data preparation in the pipeline. It acts as a specialized transformer, converting the human-readable, context-rich Silver `features` dataset into a purely numerical, normalized, and standardized Parquet file that is perfectly optimized for machine learning.

Its sole purpose is to "translate" market context into the mathematical language that ML models understand, ensuring the data is presented in a way that maximizes the model's ability to learn meaningful patterns.

## Core Purpose

Machine learning models, particularly tree-based models like the one used in the Platinum layer, struggle with raw price data and categorical text. The Gold Layer solves this by performing several key transformations that make the data **scale-invariant**, **time-invariant**, and **purely numerical**. This preprocessing step is not just about cleaning data; it is a form of deliberate feature engineering that is critical for the success of the entire discovery process.

## How It Works

The script ingests the feature file from the Silver layer and applies a sequence of non-destructive transformations.

1.  **Data Ingestion**: The script reads the comprehensive market feature file from `silver_data/features/{instrument}.parquet`.

2.  **Relational Transformation**: This is a critical step to make features comparable across different price levels and time periods.

    - **Problem**: An SMA value of 1.0850 for EURUSD is meaningless to a model without the context of the current price. Is that far away or close?
    - **Solution**: The script identifies all columns representing an absolute price level (OHLC, moving averages, Bollinger Bands, support/resistance, etc.). It then converts each of these into a normalized distance from the current closing price. For example, the `SMA_50` column is replaced by `SMA_50_dist_norm`, calculated as `(SMA_50_price - close_price) / close_price`. This new feature represents "how far away, as a percentage, is the 50-period SMA from the current close?". This makes the feature independent of the absolute price.

3.  **Categorical Encoding (One-Hot Encoding)**:

    - **Problem**: A model cannot interpret text values like 'London' or 'trend'.
    - **Solution**: The script identifies all categorical columns (`session`, `trend_regime`, `vol_regime`) and converts them into a series of binary (0/1) columns using one-hot encoding. For instance, the `session` column is replaced by several new columns like `session_London`, `session_New_York`, etc. A candle that occurred during the London session will have a `1` in the `session_London` column and a `0` in all others.

4.  **Candlestick Pattern Compression**:

    - **Problem**: The `talib` library outputs candlestick pattern scores on a noisy scale (e.g., -200, -100, 0, 100, 200), which can be difficult for a model to interpret consistently.
    - **Solution**: The script bins these scores into a simple, discrete 5-point scale: Strong Bearish (-1.0), Weak Bearish (-0.5), Neutral (0.0), Weak Bullish (0.5), and Strong Bullish (1.0). This reduces noise and simplifies the feature space, making the patterns more impactful.

5.  **Time-Series Standardization (Scaling)**:
    - **Problem**: Features like RSI (0-100 scale) and ATR (an absolute price value) exist on vastly different scales. This can cause models to incorrectly assign more importance to features with larger numerical ranges. Furthermore, scaling time-series data incorrectly can introduce **look-ahead bias**, where information from the future is used to scale data in the past, invalidating the model.
    - **Solution**: The script applies a **rolling window standardization**. For each data point in a column, it calculates the mean and standard deviation of a _preceding window of data points_ (e.g., the last 200 candles) and uses these statistics to scale the current data point. This ensures that the scaling process is time-series-aware and **100% free of look-ahead bias**, which is a methodologically critical step for any financial modeling.

## Key Architectural Features

- **Prevention of Look-Ahead Bias**: The use of a rolling window for feature scaling is the single most important architectural feature of this layer. It guarantees that the preprocessing is methodologically sound and that the resulting models are not contaminated with future information.
- **Feature Engineering for ML**: Each transformation is a deliberate choice to engineer features that are more learnable. Relational transformation creates scale-invariance, and one-hot encoding provides clear, numerical inputs.
- **Idempotent & Stateless**: The script is stateless; it processes one file at a time and its output depends only on its input and the configuration. Running it multiple times on the same input will always produce the same output.
- **Efficiency**: Continues the use of Parquet for fast I/O and downcasts data types (`downcast_dtypes`) to optimize memory usage and file size.

## Inputs and Outputs

- **Input**: A single `.parquet` feature file from `silver_data/features/`.

  - **Example Filename**: `EURUSD60.parquet`

- **Output**: A single `.parquet` ML-ready feature file in `gold_data/features/`.
  - **Example Filename**: `EURUSD60.parquet`
  - **Schema Transformation**: The output schema is purely numerical and ready for a machine learning model.
    - `SMA_50` -> `SMA_50_dist_norm`
    - `session` -> `session_London`, `session_New_York`, `session_Tokyo`, etc.
    - `RSI_14` -> `RSI_14` (but now scaled to have a rolling mean of ~0 and std dev of ~1)
    - `CDLHAMMER` -> `CDLHAMMER` (but now with values like -1.0, 0.0, 0.5, 1.0)
    - The original `time` column is preserved for joining purposes but all other non-numeric columns are removed.

## Configuration

The behavior of the Gold layer is primarily controlled by one critical parameter in `config.py`.

- `GOLD_SCALER_ROLLING_WINDOW`: This integer defines the size of the lookback window used for the time-series-aware standardization. A larger window provides a more stable mean/std for scaling but is slower to react to changes in market volatility. A smaller window is more adaptive but can be noisier. The default of `200` is a common and sensible starting point.

## How to Run the Script

1.  **Via the Master Orchestrator (Recommended)**:
    The `orchestrator.py` script will run this layer automatically after the Silver layer is complete.

    ```bash
    python orchestrator.py
    ```

2.  **Standalone (Interactive Mode)**:
    Run the script directly. It will scan for new `.parquet` feature files in `silver_data/features` and present an interactive menu to choose which instrument(s) to process.

    ```bash
    python scripts/gold_data_generator.py
    ```

3.  **Standalone (Targeted Mode)**:
    Run the script with a specific Silver feature filename as a command-line argument.
    ```bash
    python scripts/gold_data_generator.py EURUSD60.parquet
    ```
