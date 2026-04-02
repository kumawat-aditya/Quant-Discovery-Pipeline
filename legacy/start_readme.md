# [INFO] The Pipeline Orchestrator (`orchestrator.py`)

This script is the master controller for the entire Strategy Finder project. It provides a simple, powerful interface to run the complete, end-to-end data processing and strategy discovery pipeline for a single instrument with just one command.

Its purpose is to **fully automate the workflow**, from raw data to final, validated strategies, eliminating the need for any human intervention after the initial setup.

## What It Is

The Orchestrator is a Python script that acts as a "master conductor." It executes each of the individual layer scripts (`bronze_data_generator.py`, `silver_data_generator.py`, etc.) in the correct sequence. It intelligently passes the necessary command-line arguments to each script to ensure the entire run is focused on a single, user-selected market.

This script makes it possible to launch a full discovery and validation run, walk away, and return to find the final reports and trade logs ready for analysis in the dashboard.

## How It Works

The script follows a simple but robust sequence of operations:

1.  **Market Selection:** Upon launch, the script scans the `raw_data/` directory and presents a numbered list of all available `.csv` files.
2.  **User Input:** It prompts the user to select **one** market file to process for the entire pipeline run.
3.  **Automated Execution:** Once a market is selected, the orchestrator takes over completely. It begins executing the pipeline scripts in the correct order:
    - First, it runs the `diamond_data_prepper.py` to ensure all necessary data for the validation stage is cached. It intelligently tells the prepper to only process markets that share the same timeframe as the user's selected file.
    - Next, it proceeds through the discovery layers (Bronze, Silver, Gold, Platinum), passing the selected filename as a command-line argument to each script. This ensures they run in "Targeted Mode" and only process the data for the chosen instrument.
    - Finally, it runs the validation layers (Diamond, Zircon), again passing the selected filename as an argument to make them run non-interactively.
4.  **Real-time Output:** The script streams the output from each underlying script directly to the console. This allows you to monitor the progress and see any warnings or errors in real-time.
5.  **Error Handling:** If any script in the sequence fails (i.e., exits with an error), the orchestrator immediately halts the entire pipeline and reports which stage failed. This prevents the system from continuing with incomplete or corrupt data.
6.  **Completion:** If all stages complete successfully, it prints a final success message and reminds the user of the command to launch the analysis dashboard.

## [INFO] How to Use

The script is designed to be run from the **root directory** of your project.

1.  Ensure all your raw market data `.csv` files are in the `/raw_data/` directory.
2.  Open your terminal in the project's root folder.
3.  Run the orchestrator:
    ```bash
    python orchestrator.py
    ```
4.  The script will display a list of available market files. Enter the corresponding number and press `Enter`.
    ```
    --- Select a Raw Market File for the Full Pipeline ---
      [1] EURUSD15.csv
      [2] GBPUSD15.csv
      [3] XAUUSD15.csv
    Enter the number of the file to process (1-3): 3
    ```
5.  That's it! The entire pipeline will now run automatically.

## 🛠️ Script Architecture

- **`run_script(script_name, args=None)`:** A helper function that takes a script name and optional command-line arguments. It uses Python's `subprocess` module to execute the script and robustly handles its output and any potential errors.
- **`main()`:** The main function that contains the primary control flow: market selection, defining the sequence of pipeline stages, and iterating through them.
- **`pipeline_stages`:** A list of dictionaries that defines the entire workflow. Each dictionary specifies the `name` of the script to run and the `args` (command-line arguments) to pass to it. This makes the pipeline easy to read, modify, and maintain.
