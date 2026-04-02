# Quantitative Strategy Discovery Pipeline

A high-performance, end-to-end pipeline for automated discovery of quantitative trading strategies from raw OHLCV market data. The system exhaustively simulates millions of trades, enriches them with 300+ market-context features, and feeds the result into either a **Decision Tree rule miner** or a **unified XGBoost classifier** to extract statistically-validated edges.

---

## Project Overview

The pipeline ingests raw forex OHLCV CSV files and answers the question:

> _"Given the market conditions at a specific candle, what Stop Loss / Take Profit configuration has a statistically proven edge?"_

It does this through five layered stages, culminating in one of two terminal strategies:

| Terminal          | Output                                                              |
| ----------------- | ------------------------------------------------------------------- |
| **Decision Tree** | Human-readable market condition rules saved as Parquet              |
| **XGBoost**       | Trained binary classification model (`.json`) + feature importances |

---

## Pipeline Summary

```
data/raw/{instrument}.csv
        │
        ▼
  🥉 Bronze Layer     — Exhaustive BUY/SELL simulation (Numba JIT + multiprocessing)
        │                 Output: data/bronze/{inst}.parquet
        ▼
  🥈 Silver Layer     — Feature engineering + trade enrichment (indicators, ZigZag S/R, sessions)
        │                 Output: data/silver/features/ + data/silver/chunked_outcomes/
        ▼
  🥇 Gold Layer       — ML normalisation (relational transform, one-hot, rolling Z-score)
        │                 Output: data/gold/features/{inst}.parquet
        ▼
  🔀 Platinum Layer   — SPLIT based on user selection
        │
        ├── [Decision Tree path] ─────────────────────────────────────────────────────┐
        │     data_prepper.py   → Map-Reduce blueprint discovery                       │
        │     strategy_discoverer.py → DecisionTree rule mining                        │
        │     Output: data/platinum/discovered_strategies/{inst}.parquet           ✅ STOP
        │
        └── [XGBoost path] ────────────────────────────────────────────────────────────┐
              dataset_builder.py → Gold ⋈ Silver inner join → training matrix           │
              Output: data/platinum/targets/{inst}/part_N.parquet                       │
                    │                                                                    │
                    ▼                                                                    │
            ⬡ Diamond Layer  — XGBoost training (full-memory or iterative iterator)    │
              Output: data/diamond/strategies/{inst}_xgb.json                      ✅ STOP
```

---

## Prerequisites

- **Python 3.9+**
- **TA-Lib C library** — must be installed before the Python wrapper:
  - **Linux**: Build from source — see [TA-Lib GitHub](https://github.com/TA-Lib/ta-lib)
  - **macOS**: `brew install ta-lib`
  - **Windows**: Use the pre-built wheel in the repo root (`ta_lib-0.6.7-cp313-cp313-win_amd64.whl`)

---

## Setup

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd Quant-Strategy-discovery

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install TA-Lib C library first (Linux example)
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
./configure --prefix=/usr && make && sudo make install && cd ..

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Place raw OHLCV CSV files in data/raw/
#    Filenames must follow the pattern {INSTRUMENT}{TIMEFRAME}.csv
#    e.g.  EURUSD1.csv  EURUSD15.csv  XAUUSD60.csv
```

### Key Dependencies

| Package        | Purpose                                            |
| -------------- | -------------------------------------------------- |
| `numba`        | JIT-compile the trade simulation inner loop        |
| `ta`           | Technical indicator calculations                   |
| `TA-Lib`       | Candlestick pattern recognition (CDL functions)    |
| `pyarrow`      | Parquet I/O throughout the pipeline                |
| `scikit-learn` | `DecisionTreeRegressor` for the Decision Tree path |
| `xgboost`      | Model training in the Diamond layer                |
| `tqdm`         | Progress bars                                      |

---

## Running the Pipeline

### Option A — Orchestrator (Recommended)

The orchestrator runs the full pipeline non-interactively after an initial two-question setup:

```bash
python orchestrator.py
```

1. Select the instrument (lists all CSVs in `data/raw/`)
2. Select pipeline type: `[1] Decision Tree` or `[2] XGBoost`

The orchestrator then runs Bronze → Silver → Gold → (Platinum Decision Tree → STOP) or (Platinum Dataset Builder → Diamond → STOP).

### Option B — Layer by Layer

Each layer script can be run independently. Pass the instrument name (without extension) as the first argument to skip interactive prompts:

```bash
# Bronze — simulate trades
python src/layers/bronze/generator.py EURUSD60

# Silver — feature engineering + enrichment
python src/layers/silver/generator.py EURUSD60

# Gold — normalise features
python src/layers/gold/generator.py EURUSD60

# Platinum — Decision Tree path
python src/layers/platinum/data_prepper.py EURUSD60
python src/layers/platinum/strategy_discoverer.py EURUSD60

# OR — XGBoost path
python src/layers/platinum/dataset_builder.py EURUSD60
python src/layers/diamond/trainer.py EURUSD60
```

If no argument is given, each script will present an interactive file selection menu.

---

## Configuration

All pipeline parameters are stored in `config/config.py`. No other file needs to be modified for a standard run.

| Section      | Key Parameters                                                                              |
| ------------ | ------------------------------------------------------------------------------------------- |
| **Global**   | `MAX_CPU_USAGE`, log levels                                                                 |
| **Bronze**   | `TIMEFRAME_PRESETS` (SL/TP grids per timeframe), `BRONZE_GENERATION_MODE`, spread maps      |
| **Silver**   | Indicator periods (`SMA_PERIODS`, `EMA_PERIODS`, …), `SWING_ATR_MULTIPLIER`, session bins   |
| **Gold**     | `GOLD_NORMALIZATION_CONFIG` (relational transform rules), `GOLD_SCALER_ROLLING_WINDOW`      |
| **Platinum** | `PLATINUM_DT_MAX_DEPTH`, `PLATINUM_MIN_CANDLES_PER_RULE`, `PLATINUM_DENSITY_LIFT_THRESHOLD` |
| **Diamond**  | `DIAMOND_XGB_PARAMS`, `DIAMOND_BOOST_ROUNDS`, `DIAMOND_LOAD_FULL_DATASET_IN_MEMORY`         |

Set `DIAMOND_LOAD_FULL_DATASET_IN_MEMORY = True` if you have ≥ 64 GB RAM for significantly faster XGBoost training.

---

## Directory Structure

```
Quant-Strategy-discovery/
├── orchestrator.py                    ← Single entry point — option-driven pipeline runner
├── config/
│   └── config.py                      ← All pipeline parameters
├── src/
│   ├── layers/
│   │   ├── bronze/
│   │   │   └── generator.py           ← Trade simulation engine
│   │   ├── silver/
│   │   │   └── generator.py           ← Feature engineering + enrichment
│   │   ├── gold/
│   │   │   └── generator.py           ← ML normalisation preprocessor
│   │   ├── platinum/
│   │   │   ├── data_prepper.py        ← [DT path] Map-Reduce blueprint discovery
│   │   │   ├── strategy_discoverer.py ← [DT path] Decision Tree rule miner
│   │   │   └── dataset_builder.py     ← [XGBoost path] Training matrix builder
│   │   └── diamond/
│   │       └── trainer.py             ← [XGBoost path] XGBoost model trainer
│   └── utils/
│       ├── paths.py                   ← Centralised Path objects + ensure_directories()
│       ├── logger.py                  ← Rotating file + console logger setup
│       ├── file_selector.py           ← scan_new_files / select_files_interactively
│       └── raw_data_loader.py         ← Clean CSV loader
├── data/
│   ├── raw/                           ← INPUT: OHLCV CSV files
│   ├── bronze/                        ← Trade simulation Parquet
│   ├── silver/
│   │   ├── features/                  ← Per-candle indicator features
│   │   └── chunked_outcomes/          ← Enriched trade chunks
│   ├── gold/
│   │   └── features/                  ← ML-normalised feature matrix
│   ├── platinum/
│   │   ├── combinations/              ← Blueprint catalogue
│   │   ├── targets/                   ← DT: density time-series │   │   │                                  XGBoost: training matrix shards
│   │   ├── discovered_strategies/     ← [DT] Human-readable rules  ← OUTPUT [DT]
│   │   └── discovery_log/             ← Incremental processing log
│   └── diamond/
│       └── strategies/                ← [XGBoost] Model JSON + feature importances  ← OUTPUT [XGBoost]
├── docs/
│   ├── bronze_architecture.md
│   ├── silver_architecture.md
│   ├── gold_architecture.md
│   ├── platinum_decision_tree_arch.md
│   ├── platinum_xgboost_arch.md
│   ├── diamond_architecture.md
│   └── FULL_ARCHITECTURE.md           ← Complete system architecture reference
├── logs/                              ← Rotating log files per layer
├── requirements.txt
└── pyenv_setup.txt                    ← Virtual environment setup notes
```

---

## Documentation

Detailed architecture documents for each layer are in the `docs/` folder:

| Document                                                                   | Covers                                                                    |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [docs/bronze_architecture.md](docs/bronze_architecture.md)                 | Trade simulation, Numba JIT, generation modes                             |
| [docs/silver_architecture.md](docs/silver_architecture.md)                 | Indicator calculation, ZigZag S/R, ATR-normalised feature enrichment      |
| [docs/gold_architecture.md](docs/gold_architecture.md)                     | Relational transform, categorical encoding, strict Z-score policy         |
| [docs/platinum_decision_tree_arch.md](docs/platinum_decision_tree_arch.md) | Map-Reduce blueprint discovery, Decision Tree rule mining (terminal path) |
| [docs/platinum_xgboost_arch.md](docs/platinum_xgboost_arch.md)             | Gold⋈Silver join, training matrix construction (transitional to Diamond)  |
| [docs/diamond_architecture.md](docs/diamond_architecture.md)               | XGBoost training, full-memory vs iterative mode, temporal split           |
| [docs/FULL_ARCHITECTURE.md](docs/FULL_ARCHITECTURE.md)                     | Complete system overview, all layer flows, config reference, ADRs         |
