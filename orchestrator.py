# orchestrator.py (V3.2 - The Master Conductor with TQDM Fix)

"""
The Pipeline Orchestrator

This script is the master controller for the entire Strategy Finder project. It
provides a simple, powerful interface to run the complete, end-to-end data
processing and strategy discovery pipeline for a single instrument with just
one command.

Its purpose is to fully automate the workflow, from raw data to final,
validated strategies, eliminating the need for any human intervention after the
initial setup.
"""

import os
import sys
import subprocess
from time import sleep

# --- ANSI Color Codes for Better Terminal Output ---
class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'raw_data')


def run_script(script_name: str, args: list = None):
    """
    Executes a script from the 'scripts' directory, streaming its output in real-time.
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"{colors.RED}{colors.BOLD}[FATAL] Script not found: {script_path}{colors.ENDC}")
        return False

    # ### <<< CRITICAL FIX: Added the '-u' flag for unbuffered output >>>
    # This forces the child script's Python interpreter to send output immediately,
    # which is essential for rendering tqdm progress bars correctly.
    command = [sys.executable, '-u', script_path]
    if args:
        command.extend(args)

    print(f"\n{colors.HEADER}{'='*25} EXECUTING: {script_name} {' '.join(args or [])} {'='*25}{colors.ENDC}")
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        
        # This line-by-line reading now works correctly because of the '-u' flag.
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        print(f"\n{colors.GREEN}{colors.BOLD}[SUCCESS] Stage '{script_name}' completed.{colors.ENDC}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n{colors.RED}{colors.BOLD}[FAILED] Stage '{script_name}' exited with error code {e.returncode}.{colors.ENDC}")
        return False
    except Exception as e:
        print(f"\n{colors.RED}{colors.BOLD}[ERROR] An unexpected error occurred while running {script_name}: {e}{colors.ENDC}")
        return False


def main():
    """The main function to orchestrate the entire pipeline."""
    print(f"{colors.BOLD}{colors.BLUE}===== Starting the Full Strategy Discovery Pipeline ====={colors.ENDC}")

    try:
        raw_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')])
        if not raw_files:
            print(f"{colors.RED}[ERROR] No raw data files (.csv) found in '{RAW_DATA_DIR}'. Exiting.{colors.ENDC}")
            return
    except FileNotFoundError:
        print(f"{colors.RED}[ERROR] Raw data directory not found at '{RAW_DATA_DIR}'. Exiting.{colors.ENDC}")
        return

    print(f"\n{colors.YELLOW}--- Select a Master Instrument for the Full Pipeline ---{colors.ENDC}")
    for i, f in enumerate(raw_files):
        print(f"  [{i+1}] {f}")
    
    try:
        choice = int(input(f"Enter the number of the file to process (1-{len(raw_files)}): ")) - 1
        if not 0 <= choice < len(raw_files): raise ValueError
        
        raw_csv_file = raw_files[choice]
        instrument_name = os.path.splitext(raw_csv_file)[0]
        bronze_parquet_file = f"{instrument_name}.parquet"
        silver_parquet_file = f"{instrument_name}.parquet"

        print(f"{colors.CYAN}[INFO] You selected: {raw_csv_file}. The pipeline will now run non-interactively.{colors.ENDC}")
    
    except (ValueError, IndexError):
        print(f"{colors.RED}[ERROR] Invalid selection. Exiting.{colors.ENDC}")
        return
    
    # --- The Final, Corrected Pipeline Sequence ---
    pipeline_stages = [
        {"name": "bronze_data_generator.py", "args": [raw_csv_file]},
        {"name": "silver_data_generator.py", "args": [bronze_parquet_file]},
        {"name": "gold_data_generator.py", "args": [silver_parquet_file]},
        {"name": "platinum_data_prepper.py", "args": [instrument_name]},
        {"name": "platinum_strategy_discoverer.py", "args": [instrument_name]},
        {"name": "diamond_data_prepper.py", "args": [instrument_name]},
        {"name": "diamond_backtester.py", "args": [instrument_name]},
        {"name": "diamond_validator.py", "args": [instrument_name]},
    ]

    for stage in pipeline_stages:
        sleep(1) 
        if not run_script(stage["name"], args=stage.get("args")):
            print(f"\n{colors.RED}{colors.BOLD}PIPELINE HALTED due to an error in stage: {stage['name']}.{colors.ENDC}")
            print(f"{colors.YELLOW}Please review the log files and the output above to diagnose the issue.{colors.ENDC}")
            return
            
    print(f"\n{colors.GREEN}{colors.BOLD}>>>>> Full pipeline completed successfully for {instrument_name}! <<<<<{colors.ENDC}")
    print(f"{colors.YELLOW}You can now explore the final reports in the 'diamond_data/final_reports' directory.{colors.ENDC}")

if __name__ == "__main__":
    main()