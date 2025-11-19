### **Deployment Guide: MT5 Live Strategy Validator on Windows**

This guide will walk you through the entire process of setting up the environment, configuring the application, and running the live validator on your Windows machine.

#### **Prerequisites**

1.  **Windows OS:** You must be on a Windows machine.
2.  **MetaTrader 5 Terminal:** You must have the MT5 desktop application installed from your broker.
3.  **Python 3.9+:** Ensure you have a modern version of Python installed. You can get it from the [official Python website](https://www.python.org/downloads/). During installation, make sure to check the box that says **"Add Python to PATH"**.
4.  **Completed Research Pipeline:** You must have already run the research pipeline (`orchestrator.py`) for at least one instrument. This guide assumes the output files (e.g., for `EURUSD60`) exist in the `diamond_data` and `platinum_data` directories.

---

### **Step 1: Project Setup and Environment**

First, we'll set up the project folder and a dedicated Python environment.

1.  **Create a New Folder:** Create a new folder for your project, for example, `C:\StrategyFinder`.

2.  **Organize Files:** Inside `C:\StrategyFinder`, create the directory structure we designed. Copy all your existing research scripts and the new MT5 validator scripts into the correct locations. Your folder should look like this:

    ```
    C:\StrategyFinder\
    ├── bronze_data\
    ├── diamond_data\
    │   └── master_reports\
    │       └── EURUSD60.parquet  (Example output from research)
    ├── gold_data\
    ├── logs\
    ├── platinum_data\
    │   └── combinations\
    │       └── EURUSD60.parquet  (Example output from research)
    ├── raw_data\
    ├── silver_data\
    ├── scripts/
    │   └── ... (all research scripts)
    ├── mt5_validator/
    │   ├── logs/
    │   ├── live_config.py
    │   ├── mt5_connector.py
    │   ├── data_engine.py
    │   ├── feature_engine.py
    │   ├── strategy_loader.py
    │   ├── trading_engine.py
    │   └── mt5_validator.py
    ├── config.py
    ├── orchestrator.py
    └── requirements.txt
    ```

3.  **Open Command Prompt:** Open the Windows Command Prompt (or PowerShell). You can do this by typing `cmd` in the Start Menu.

4.  **Navigate to Project Folder:** Use the `cd` command to navigate to your project's root directory.

    ```cmd
    cd C:\StrategyFinder
    ```

5.  **Create Virtual Environment:** Create a dedicated Python environment for this project. This isolates its dependencies from your global Python installation.

    ```cmd
    python -m venv venv
    ```

6.  **Activate Virtual Environment:** You must activate the environment in every new terminal session before running the scripts.

    ```cmd
    .\venv\Scripts\activate
    ```

    You will know it's active because your command prompt will be prefixed with `(venv)`.

7.  **Install Dependencies:** Install all the necessary Python libraries using the `requirements.txt` file.
    ```cmd
    pip install -r requirements.txt
    ```
    You will also need the `MetaTrader5` library specifically for the validator.
    ```cmd
    pip install MetaTrader5
    ```

---

### **Step 2: Configuration**

Now, we need to edit `live_config.py` to match your specific setup.

1.  **Find your MT5 Path:**

    - Find the shortcut for your MetaTrader 5 terminal on your desktop.
    - Right-click it and select **"Properties"**.
    - In the "Target" field, you will see the full path. It will look something like `"C:\Program Files\Your Broker - MetaTrader 5\terminal64.exe"`.
    - Copy this full path.

2.  **Edit `live_config.py`:**

    - Open the file `C:\StrategyFinder\mt5_validator\live_config.py` in a text editor (like VS Code, Notepad++, etc.).
    - **Update `MT5_PATH`:** Paste the path you copied. Remember to use double backslashes (`\\`) in Python strings.
    - **Update Credentials:** Fill in your demo account `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER`.
    - **Verify Symbols:** Check the `SYMBOLS_TO_TRACK` list. Make sure the symbol names (`"EURUSD"`, `"GBPUSD"`) exactly match what you see in the "Market Watch" window in your MT5 terminal.
    - **Verify Timeframe:** Ensure the `TIMEFRAME` variable matches the timeframe of the strategies you want to test (e.g., `mt5.TIMEFRAME_H1` for strategies discovered on a 60-minute chart).

    **Example `live_config.py` section:**

    ```python
    # Example for a standard Windows installation:
    MT5_PATH: str = "C:\\Program Files\\Your Broker - MetaTrader 5\\terminal64.exe"

    # Your MT5 account credentials.
    MT5_LOGIN: int = 87654321
    MT5_PASSWORD: str = "MySecurePassword123"
    MT5_SERVER: str = "YourBroker-Demo"
    ```

3.  **Create Log Directory:** Inside the `mt5_validator` folder, manually create an empty folder named `logs`. The script needs this folder to exist to write its log file.

---

### **Step 3: Running the Application**

With setup and configuration complete, you are ready to run the validator.

1.  **Launch and Log In to MT5:**

    - Start your MetaTrader 5 terminal application.
    - Log in to the **same demo account** that you configured in `live_config.py`.
    - **IMPORTANT:** Leave the terminal running. Do not close it.

2.  **Ensure Environment is Active:** In your Command Prompt, make sure you are in the project's root directory (`C:\StrategyFinder`) and that your virtual environment is active (you see `(venv)` at the start of the line).

3.  **Run the Main Script:** Execute the main validator script from the root directory.

    ```cmd
    python mt5_validator/mt5_validator.py
    ```

4.  **Monitor the Output:**

    - The script will start logging its progress directly to your console.
    - You should see messages like:
      - `Initializing MT5 Live Strategy Validator...`
      - `Attempting to initialize MetaTrader 5 terminal...`
      - `Successfully connected to account...`
      - `Loading master strategies for EURUSDH1...`
      - `Prefilling historical data cache...`
      - `--- Validator Startup Complete. Entering Main Loop. ---`
    - The application is now running. It will be quiet until a new trading signal is detected.

5.  **Check for Trades:** When a strategy's conditions are met on a newly closed candle, you will see a `TRADE SIGNAL DETECTED` message in the log, followed by the execution result. You will see the new trade appear in the "Trade" tab of your MT5 Terminal. **Crucially, check the "Comment" column of that trade—it should match the `trigger_key` of the strategy.**

6.  **Stopping the Bot:** To stop the application, go to the command prompt window where it is running and press **`Ctrl + C`**. The script will detect this, log a "Shutting down..." message, and disconnect cleanly from MT5.
