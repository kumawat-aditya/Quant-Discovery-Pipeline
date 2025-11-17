# Platinum Layer: The Rule Miner & Strategy Discoverer

The Platinum Layer is the intelligent heart of the discovery pipeline. It consists of two distinct but cooperative scripts that work together to transform the billions of data points from the previous layers into a finite set of explicit, human-readable trading strategies. This layer is where abstract patterns are identified and the specific market conditions for trading them are mined using machine learning.

The process is split into two major parts to handle the immense scale of the data efficiently:

1.  **`platinum_prepper.py`**: A high-performance data aggregator that discovers abstract strategy "blueprints".
2.  **`platinum_strategy_discoverer.py`**: A machine learning engine that finds the specific market rules for trading those blueprints.

---

## Part 1: `platinum_prepper.py` - The Blueprint Discoverer

### Core Purpose

This script's job is to make sense of the billions of enriched trades from the Silver Layer. It sifts through this massive dataset to find and categorize recurring, high-level strategy patterns, which we call **"blueprints."** A blueprint is an abstract rule about trade structure, such as:

- _"Place the Stop-Loss 20% of the way to the support line and use a fixed 3:1 reward/risk ratio."_
- _"Place the Take-Profit 5 basis points above the upper Bollinger Band and use a fixed 0.5% Stop-Loss."_

After discovering all unique blueprints, it performs a massive aggregation to count how many winning trades for each blueprint occurred on every single candle. This pre-computation is essential for the machine learning stage that follows.

### How It Works

This script is architected as a high-throughput **Map-Reduce** pipeline to process data that is too large to fit into memory.

1.  **Binning**: The script first reads the continuous relational features from the Silver layer chunks (e.g., `sl_place_pct_to_resistance = 0.83`) and "bins" them into discrete categories (e.g., `sl_place_pct_to_resistance_bin = 8`). This discretization is what defines the abstract blueprints.

2.  **Phase 1: Map (Parallel Discovery)**:

    - Multiple worker processes read the Silver `chunked_outcomes` files in parallel.
    - Each worker discovers all unique blueprints within its assigned chunk and aggregates the trade counts for each blueprint per candle.
    - The results are streamed to a large in-memory buffer in the main process.

3.  **Phase 2: Shuffle (Sharded Streaming)**:

    - This is the key optimization to avoid I/O bottlenecks. When the memory buffer reaches a size threshold, it is "flushed" to disk.
    - Instead of writing to thousands of tiny files, the buffer's contents are hashed and appended to a small, fixed number of temporary "shard" files. This concentrates all disk writes, making the process much faster.

4.  **Phase 3: Reduce (Parallel Consolidation)**:
    - After all chunks have been processed, a final parallel process begins.
    - Each worker is assigned one temporary shard file. It loads its shard, performs a final aggregation (`groupby('key')`), and writes the final, clean target files—one small Parquet file for each unique strategy blueprint.

### Inputs and Outputs (Preprocessor)

- **Input**: The directory of chunked Parquet files from the Silver layer: `silver_data/chunked_outcomes/{instrument}/`.

- **Outputs**:
  1.  `platinum_data/combinations/{instrument}.parquet`: The master list of all discovered blueprints. Each blueprint is assigned a unique hash `key` for identification and its definition is stored (e.g., type, SL definition, TP definition).
  2.  `platinum_data/targets/{instrument}/`: A directory containing potentially thousands of small Parquet files. Each file (`{key}.parquet`) contains the performance data (`entry_time`, `trade_count`) for a single blueprint.

---

## Part 2: `platinum_strategy_discoverer.py` - The Rule Miner

### Core Purpose

This script is the machine learning engine. It takes the blueprints discovered by the preprocessor and asks the critical question: **"What were the market conditions (from the Gold data) when this blueprint was _unusually_ successful?"** It uses a Decision Tree model to find the answer, which it extracts as a human-readable rule. The final output is a complete strategy, which is the combination of a **blueprint** (the trade structure) and a **market rule** (the entry condition).

### How It Works

1.  **Data Loading**: The script loads the ML-ready Gold features dataset into memory, where it is shared efficiently with all worker processes. It also loads the list of blueprints to be analyzed from the `combinations` file.

2.  **Training Data Assembly**: For each blueprint, the script performs the following:

    - It loads the corresponding small `target` file (containing `entry_time` and `trade_count`).
    - It merges this target data with the Gold features on the `time` column. This creates a unique training dataset for each blueprint.

3.  **The Feedback Loop (Blacklisting)**: Before training, the script checks if a `blacklist` file exists for the instrument. This file is created by the Diamond backtester and contains strategies that have already been tested and failed.

    - If a rule for the current blueprint is on the blacklist, the script "prunes" the training data by setting the `trade_count` to zero for all candles matching the failed rule.
    - This powerful mechanism forces the Decision Tree to ignore previously failed patterns and find **novel, alternative rules**, allowing the system to learn from its mistakes.

4.  **Decision Tree Training**: A `DecisionTreeRegressor` model is trained.

    - The features (`X`) are the Gold market data (e.g., scaled RSI, session dummies).
    - The target (`y`) is the `trade_count` for the blueprint.
    - The model's goal is to find paths (combinations of features) that lead to "leaves" with a high average `trade_count`.

5.  **Rule Extraction & Filtering**: The script traverses the trained tree and translates the paths to the highest-performing leaves into human-readable rule strings (e.g., `RSI_14_scaled > 1.5 and session_London == 1`). It then applies strict quality filters from `config.py` to discard rules that are based on too few samples or don't provide a significant performance "lift" over the baseline.

### Inputs and Outputs (Discoverer)

- **Inputs**:

  - `gold_data/features/{instrument}.parquet`: The source of ML-ready market conditions.
  - The outputs of the Platinum Preprocessor (`combinations` and `targets` directories).
  - `platinum_data/blacklists/{instrument}.parquet`: (Optional) The feedback file from the Diamond backtester.

- **Outputs**:
  - `platinum_data/discovered_strategies/{instrument}.parquet`: The primary output of the Platinum Layer. A single file containing a list of complete strategies, each defined by a blueprint `key` and a `market_rule` string.
  - `platinum_data/exhausted_keys/{instrument}.parquet`: A list of blueprints for which the model could no longer find any new, profitable rules after considering the blacklist.
  - `platinum_data/discovery_log/`: Log files that track which blueprints have been processed, making the script fully resumable.

## Configuration

The Platinum Layer's behavior is tuned via `config.py`:

- **Preprocessor Settings**:
  - `PLATINUM_BPS_BIN_SIZE`: The size of the bin for discretizing relational features.
  - `PLATINUM_BUFFER_FLUSH_THRESHOLD`: The memory buffer size before flushing to shards.
  - `PLATINUM_NUM_SHARDS`: The number of temporary files to use in the shuffle phase.
- **Discoverer Settings**:
  - `PLATINUM_MIN_CANDLE_LIMIT`: The minimum number of times a blueprint must have occurred to be considered.
  - `PLATINUM_DT_MAX_DEPTH`: The maximum depth of the Decision Tree, controlling rule complexity.
  - `PLATINUM_MIN_CANDLES_PER_RULE`: A key filter to prevent rules based on statistically insignificant sample sizes.
  - `PLATINUM_DENSITY_LIFT_THRESHOLD`: Ensures a rule provides a significant performance edge over the blueprint's baseline.

## How to Run the Scripts

Both scripts are designed to be run in sequence for a given instrument.

1.  **Via the Master Orchestrator (Recommended)**:
    The `orchestrator.py` will run both scripts in the correct order automatically.

    ```bash
    python orchestrator.py
    ```

2.  **Standalone (Targeted Mode)**:
    You must run the preprocessor first, followed by the discoverer, targeting the same instrument name.

    ```bash
    # Step 1: Run the preprocessor
    python scripts/platinum_prepper.py EURUSD60

    # Step 2: Run the discoverer
    python scripts/platinum_strategy_discoverer.py EURUSD60
    ```
