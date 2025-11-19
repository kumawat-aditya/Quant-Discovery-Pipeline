# About the MT5 Live Strategy Validator

This document provides a detailed explanation of the architecture and core components of the MT5 Live Strategy Validator. This application serves as the critical bridge between the theoretical strategies discovered in the Python research environment and the practical realities of a live trading platform.

The system is designed with a modular, object-oriented approach, where each component has a single, clear responsibility. This separation of concerns makes the application robust, maintainable, and easier to debug.

---

### **Component: `mt5_connector.py`**

**Purpose:** To manage the connection to the MetaTrader 5 terminal with stability and grace.

- **Encapsulation:** All low-level connection logic is contained within the `MT5Connector` class. This abstracts away the details of `mt5.initialize()` and `mt5.shutdown()`, allowing the main application to use a simple, high-level interface.
- **Centralized Configuration:** The connector directly imports credentials and the terminal path from `live_config.py`, ensuring a single source of truth for all connection parameters.
- **Core `connect()` Method:** This is the heart of the class. It calls `mt5.initialize()` to establish the vital link to the running terminal application. It performs crucial error checks after both initialization and login to validate the connection status.
- **Graceful `disconnect()` Method:** This method properly terminates the connection using `mt5.shutdown()`. This is essential for releasing system resources and preventing orphaned connections.
- **Robust Resource Management:** The class implements Python's **Context Manager** protocol (`__enter__`, `__exit__`). This allows the use of a `with` statement, which is the gold standard for resource handling. It guarantees that `disconnect()` is called automatically when the application exits, even if an unexpected error occurs, preventing instability.
- **Standalone Testability:** A dedicated `if __name__ == "__main__":` block allows this file to be run directly. This provides a quick and isolated way to test if your credentials and terminal path are configured correctly, greatly simplifying initial setup and debugging.

---

### **Component: `data_engine.py`**

**Purpose:** To provide a continuous and efficient stream of up-to-date market data.

- **Constructor (`__init__`)**: The constructor initializes an in-memory dictionary, `_data_cache`, which will hold a separate Pandas DataFrame of OHLC data for each tracked symbol.
- **`prefill_cache()` Method:** This is a critical, one-time operation performed at startup. It populates the cache with a substantial amount of historical data using `mt5.copy_rates_from_pos`. Without this initial data block, the feature indicators would have no historical context to be calculated on.
- **Efficient `update_data()` Method:** This is the core of the engine's performance optimization. Instead of re-downloading hundreds of bars on every cycle, it intelligently fetches only the _new_ candle data that has formed since its last check. This is achieved by tracking the timestamp of the last known candle and using `mt5.copy_rates_from`. This design dramatically reduces network traffic and processing load.
- **`get_dataframe()` Method:** This is the public interface for other modules. It provides the latest, complete DataFrame for a given symbol in the exact format required by the research pipeline (`time` as a column), ensuring data consistency.
- **Standalone Testability:** Like the connector, this module can be run directly to test its lifecycle: `initialize -> prefill -> wait -> update`. This allows for isolated verification of its caching and data retrieval logic.

---

### **Component: `strategy_loader.py`**

**Purpose:** To load, merge, and prepare the "Master Strategies" from the research pipeline's output files.

- **Clear Responsibility:** This module abstracts away all the file I/O and data manipulation logic required to prepare strategies for testing. The main application simply calls a single method (`load_master_strategies`) to get a clean, ready-to-use list.
- **The Critical Merge:** The most important function of this loader is its use of `pd.merge()`. It combines two key pieces of information:
  1.  From **`diamond_data/master_reports`**: The list of strategies that passed validation, identified by their `key` and `market_rule`.
  2.  From **`platinum_data/combinations`**: The blueprint definitions (`sl_def`, `tp_def`, `trade_type`) associated with each `key`.
      This merge creates a complete "strategy object" that contains both the _entry conditions_ and the _trade management rules_.
- **Standardized Data Format:** The loader returns a `List[Dict[str, Any]]`, a standard and highly convenient format. The main application loop can iterate through this list, with each item being a self-contained dictionary holding all the information needed to evaluate and execute a strategy.
- **Robustness:** The function includes checks for file existence and logs clear errors if the necessary input files from the research pipeline are missing, preventing the application from starting with incomplete data.
- **Standalone Testability:** This module can be run directly to confirm it can correctly find and parse the output files from your research pipeline for any given instrument.

---

### **Component: `trading_engine.py`**

**Purpose:** To translate abstract trading signals into concrete trade execution orders.

- **Logic Replication:** The `_calculate_sl_tp_prices` method is a careful and direct adaptation of the logic from the research `simulation_engine.py`. This ensures that the way stop-loss and take-profit levels are calculated in the live environment is **identical** to the backtest, a non-negotiable requirement for valid forward-testing.
- **Safety First Risk Management:** The `execute_trade` method's first action is to query for existing open positions for the _same strategy_. This check, governed by `MAX_OPEN_TRADES_PER_STRATEGY`, is a fundamental safety net that prevents a single overactive signal from flooding the account with duplicate trades.
- **Performance Tracking via `comment`:** The engine sets the `comment` field of every trade request to the strategy's unique `trigger_key`. This is the **most critical step for post-analysis**, as it creates an unbreakable link between an executed trade's profit or loss and the specific strategy that generated it.
- **Broker-Specific Precision:** The engine correctly uses `mt5.symbol_info()` to fetch real-time data like the current bid/ask spread and the symbol's required price precision (`digits`). This is vital for constructing valid trade requests that the broker's server will accept without rejection.
- **Modularity:** This class handles all the complex details of building and sending an order. This allows the main application loop to remain clean and readable, with a simple, high-level call: `trading_engine.execute_trade(...)`.

---

### **Component: `mt5_validator.py`**

**Purpose:** To serve as the master controller, wiring all components together and running the main application loop.

- **Class-Based Structure:** The entire application is encapsulated within the `MT5Validator` class, providing an organized structure for managing state (e.g., loaded strategies) and orchestrating the workflow.
- **Clean Startup Sequence:** The `startup()` method handles the critical, one-time setup tasks: connecting to MT5, loading all strategies, and pre-filling the data cache. By separating this from the main loop, the code is cleaner, and it prevents the loop from starting if the initial setup fails.
- **The Main Event Loop (`main_loop`)**: This is the continuous heartbeat of the bot. In every cycle, it systematically:
  1.  Updates market data.
  2.  Generates the latest features for each symbol.
  3.  Evaluates every loaded strategy against the new feature set.
  4.  Executes trades on valid signals.
  5.  Waits for the configured interval before repeating.
- **Best Practice: Using the Closed Candle:** The loop deliberately uses `iloc[-2]` to get features from the last _fully formed_ candle. This is a critical design choice to prevent "repainting"—making decisions on incomplete data from the currently forming bar, which would invalidate the signals.
- **The Signal Check:** The core logic, `...query(strategy['market_rule'])`, is the moment of truth where a strategy's human-readable rule is evaluated against the live, normalized market data.
- **Graceful Shutdown:** The application is wrapped in a `try...except KeyboardInterrupt` block, ensuring that pressing `Ctrl+C` triggers a clean shutdown sequence, properly disconnecting from the MT5 terminal.
