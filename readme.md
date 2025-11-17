# Quantitative Strategy Discovery Pipeline

This project is a high-performance, end-to-end pipeline for the automated discovery, backtesting, and validation of quantitative trading strategies from raw financial market data. It employs a multi-layered data processing architecture, funneling a vast universe of potential trades through progressively sophisticated stages of enrichment, machine learning, and validation to uncover a small set of robust, high-potential strategies.

The entire system is designed to be modular, scalable, and automated, allowing for the systematic mining of market data for repeatable patterns and alpha.

## Key Features

- **Modular Architecture**: A five-layer (Bronze, Silver, Gold, Platinum, Diamond) design where each stage has a clear, distinct responsibility.
- **High Performance**: Optimized for speed using parallel processing (`multiprocessing`), JIT compilation (`Numba`), and efficient columnar data formats (`Parquet`).
- **Methodologically Sound**: Incorporates best practices to combat common pitfalls in financial ML, including time-series-aware data scaling to prevent look-ahead bias.
- **Intelligent Discovery**: Uses a Decision Tree model to mine for explicit, human-readable trading rules rather than operating as a "black box".
- **Automated Feedback Loop**: The system learns from its mistakes. Failed strategies identified during backtesting are automatically blacklisted, forcing the discovery engine to find novel, alternative rules in subsequent runs.
- **Fully Automated**: A master `orchestrator.py` script allows for a complete, end-to-end "one-command" run for any given instrument.

## Project Architecture

The pipeline processes data through a series of layers, where each layer refines and adds value to the data from the previous one.

**`raw_data/*.csv`** (Input)
↓
**1. Bronze Layer (The Possibility Engine)**: Ingests raw OHLC price data and performs a brute-force simulation to generate a massive dataset of every conceivable winning trade based on a predefined grid of Stop-Loss/Take-Profit ratios.
↓
**2. Silver Layer (The Enrichment Engine)**: Calculates a comprehensive suite of technical indicators and market features. It then enriches the Bronze layer trades with this market context, describing the relationship between SL/TP levels and key market structures.
↓
**3. Gold Layer (The ML Preprocessor)**: Transforms the human-readable Silver features into a purely numerical, scaled, and normalized format that is perfectly optimized for machine learning algorithms.
↓
**4. Platinum Layer (The Rule Miner)**: This is the intelligent core. It first discovers abstract strategy "blueprints" from the enriched trade data. Then, it uses a Decision Tree model to mine the Gold layer data for specific, human-readable market rules that predict when these blueprints are most profitable.
↓
**5. Diamond Layer (The Gauntlet)**: This is the final validation and analysis stage. It performs high-fidelity backtesting on discovered strategies, filters them through a strict set of performance criteria to find "Master Strategies," and then stress-tests these masters across a portfolio of related instruments to ensure robustness.

## Directory Structure

```
/
├── raw_data/                 # Input: Your raw OHLC price data (.csv)
├── scripts/                  # All the Python scripts for the pipeline layers
│   ├── bronze_data_generator.py
│   ├── silver_data_generator.py
│   ├── gold_data_generator.py
│   ├── platinum_prepper.py
│   ├── platinum_strategy_discoverer.py
│   ├── diamond_data_prepper.py
│   ├── diamond_backtester.py
│   ├── diamond_validator.py
│   ├── simulation_engine.py      # Shared backtesting logic
│   └── logger_setup.py         # Reusable logging utility
├── bronze_data/                # Output: Brute-force trade possibilities (.parquet)
├── silver_data/                # Output: Enriched trades and market features
├── gold_data/                  # Output: ML-ready numerical features
├── platinum_data/              # Output: Discovered blueprints and strategies
├── diamond_data/               # Output: Final reports, trade logs, and validated strategies
├── logs/                       # Output: Detailed logs for each script run
├── config.py                   # Central configuration file for all parameters
├── orchestrator.py             # Master script to run the full pipeline
└── requirements.txt            # Project dependencies
```

## Setup and Installation

### Prerequisites

- Python 3.9+
- The `TA-Lib` technical analysis library. This is a C-library and must be installed on your system _before_ you install the Python wrapper.
  - **Windows**: Download `ta-lib-0.4.0-msvc.zip` from [lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and follow instructions.
  - **macOS**: `brew install ta-lib`
  - **Linux**: Follow the instructions on the [TA-Lib GitHub](https://github.com/mrjbq7/ta-lib).

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    The project includes a `requirements.txt` file with all necessary libraries.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Add Raw Data:**
    Place your raw OHLC data in `.csv` format inside the `raw_data` directory. The files must contain columns for `time`, `open`, `high`, `low`, and `close`.

## Configuration

All pipeline parameters are managed in the central **`config.py`** file. This single source of truth allows you to tune every aspect of the discovery and validation process, including:

- **CPU Usage & Logging**: Global settings for parallelization and log levels.
- **Bronze Layer**: The SL/TP ratio grids and lookforward periods for the initial simulation.
- **Silver Layer**: The periods and parameters for all technical indicators.
- **Gold Layer**: The rolling window size for time-series-aware feature scaling.
- **Platinum Layer**: The thresholds and hyperparameters for the Decision Tree rule mining.
- **Simulation & Cost Model**: The spread, commission, and slippage assumptions for all backtests.
- **Diamond Layer**: The performance criteria (Profit Factor, Max Drawdown, etc.) for a strategy to be considered a "Master".

## How to Run the Pipeline

There are two primary ways to run the pipeline:

### 1. Fully Automated End-to-End Run (Recommended)

The `orchestrator.py` script is the master controller designed to run the entire pipeline for a single instrument without any user interaction.

1.  Make sure your desired `.csv` file is in the `raw_data` directory.
2.  Run the orchestrator:
    ```bash
    python orchestrator.py
    ```
3.  The script will prompt you to select which instrument to process. After selection, it will execute each layer in the correct sequence automatically.

### 2. Manual, Layer-by-Layer Execution

For debugging, development, or more granular control, you can run each script individually. Most scripts feature an interactive menu to select the specific file or instrument you wish to process.

**Example Sequence for `EURUSD60.csv`:**

```bash
# 1. Generate Bronze Data
python scripts/bronze_data_generator.py

# 2. Generate Silver Data
python scripts/silver_data_generator.py

# 3. Generate Gold Data
python scripts/gold_data_generator.py

# ...and so on for each layer in sequence.
```

## The Feedback Loop

A key feature of this pipeline is its ability to learn. The process works as follows:

1.  `platinum_strategy_discoverer` finds a new strategy (e.g., Blueprint A + Rule X).
2.  `diamond_backtester` tests this strategy and finds that it fails to meet the performance criteria (e.g., its profit factor is too low).
3.  The backtester adds the failing strategy (`key` + `market_rule`) to a **blacklist** file for that instrument.
4.  On the next run, the `platinum_strategy_discoverer` loads this blacklist. When it analyzes Blueprint A again, it is now forbidden from generating Rule X. This forces the Decision Tree to find a new, potentially better rule (e.g., Rule Y), thus improving the quality of discovered strategies over time.

## Outputs

The final, valuable outputs of the pipeline are located in the `diamond_data/final_reports` directory. These Parquet files provide a comprehensive, multi-faceted view of the performance of the validated "Master Strategies":

- **`{instrument}_detailed.parquet`**: Performance metrics for each strategy on every market it was tested on.
- **`{instrument}_summary.parquet`**: Average performance metrics for each strategy across all tested markets.
- **`{instrument}_regime_analysis.parquet`**: A breakdown of strategy performance under different market conditions (e.g., 'trending' vs. 'ranging', 'high-vol' vs. 'low-vol', different trading sessions).
