# orchestrator.py (V4.0 - Option-Driven Layer Pipeline Runner)

"""
The Pipeline Orchestrator

Provides a simple, interactive interface to run the full strategy discovery
pipeline for a chosen instrument.  After selecting the instrument the user
picks one of two terminal modes:

  [1] Decision Tree  →  Bronze → Silver → Gold → Platinum Data-Prepper
                                                → Platinum Strategy Discoverer  (STOP)

  [2] XGBoost        →  Bronze → Silver → Gold → Platinum Dataset Builder
                                                → Diamond Trainer               (STOP)

Every stage is executed by calling its layer script in src/layers/ directly,
so the orchestrator never needs to know the internal logic of any layer.
"""

import os
import sys
import subprocess
from pathlib import Path
from time import sleep

# ---------------------------------------------------------------------------
# ANSI Colour Codes
# ---------------------------------------------------------------------------
class colors:
    HEADER    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parent
LAYERS_DIR  = ROOT_DIR / "src" / "layers"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

# Canonical mapping: logical stage name → layer script path
LAYER_SCRIPTS = {
    "bronze":      LAYERS_DIR / "bronze"   / "generator.py",
    "silver":      LAYERS_DIR / "silver"   / "generator.py",
    "gold":        LAYERS_DIR / "gold"     / "generator.py",
    "pt_prepper":  LAYERS_DIR / "platinum" / "data_prepper.py",
    "pt_discover": LAYERS_DIR / "platinum" / "strategy_discoverer.py",
    "pt_builder":  LAYERS_DIR / "platinum" / "dataset_builder.py",
    "diamond":     LAYERS_DIR / "diamond"  / "trainer.py",
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_layer(label: str, script_path: Path, args: list = None) -> bool:
    """
    Spawns a layer script as a subprocess and streams its output line-by-line.
    The '-u' flag forces unbuffered Python output so tqdm bars render properly.
    Returns True on success, False on any non-zero exit code.
    """
    if not script_path.exists():
        print(f"{colors.RED}{colors.BOLD}[FATAL] Layer script not found: {script_path}{colors.ENDC}")
        return False

    command = [sys.executable, "-u", str(script_path)]
    if args:
        command.extend(str(a) for a in args)

    print(f"\n{colors.HEADER}{'=' * 20} EXECUTING: {label} {'=' * 20}{colors.ENDC}")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        print(f"\n{colors.GREEN}{colors.BOLD}[SUCCESS] {label} completed.{colors.ENDC}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n{colors.RED}{colors.BOLD}[FAILED] {label} exited with code {e.returncode}.{colors.ENDC}")
        return False
    except Exception as e:
        print(f"\n{colors.RED}{colors.BOLD}[ERROR] Unexpected error in {label}: {e}{colors.ENDC}")
        return False


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def _select_instrument() -> str:
    """Lists raw CSV files and returns the chosen instrument name (no extension)."""
    try:
        raw_files = sorted(f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv"))
    except FileNotFoundError:
        print(f"{colors.RED}[ERROR] Raw data directory not found: '{RAW_DATA_DIR}'{colors.ENDC}")
        sys.exit(1)

    if not raw_files:
        print(f"{colors.RED}[ERROR] No CSV files found in '{RAW_DATA_DIR}'.{colors.ENDC}")
        sys.exit(1)

    print(f"\n{colors.YELLOW}--- Select Instrument ---{colors.ENDC}")
    for i, f in enumerate(raw_files):
        print(f"  [{i + 1}] {f}")

    try:
        idx = int(input(f"\nEnter number (1-{len(raw_files)}): ").strip()) - 1
        if not 0 <= idx < len(raw_files):
            raise ValueError
        chosen = raw_files[idx]
        instrument = os.path.splitext(chosen)[0]
        print(f"{colors.CYAN}[INFO] Selected: {chosen}{colors.ENDC}")
        return instrument
    except (ValueError, IndexError):
        print(f"{colors.RED}[ERROR] Invalid selection. Exiting.{colors.ENDC}")
        sys.exit(1)


def _select_pipeline() -> str:
    """Returns 'dt' (Decision Tree) or 'xgb' (XGBoost)."""
    print(f"\n{colors.YELLOW}--- Select Pipeline Type ---{colors.ENDC}")
    print("  [1] Decision Tree  —  Platinum Data-Prepper → Strategy Discoverer  (terminal path)")
    print("  [2] XGBoost        —  Platinum Dataset Builder → Diamond Trainer    (full path)")

    choice = input("\nEnter 1 or 2: ").strip()
    if choice == "1":
        print(f"{colors.CYAN}[INFO] Pipeline: Decision Tree{colors.ENDC}")
        return "dt"
    elif choice == "2":
        print(f"{colors.CYAN}[INFO] Pipeline: XGBoost{colors.ENDC}")
        return "xgb"
    else:
        print(f"{colors.RED}[ERROR] Invalid selection. Exiting.{colors.ENDC}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"{colors.BOLD}{colors.BLUE}===== Quant Strategy Discovery Pipeline (V4.0) ====={colors.ENDC}")

    instrument = _select_instrument()
    pipeline   = _select_pipeline()

    # -----------------------------------------------------------------------
    # Build the ordered list of stages:
    #   (human-readable label, script Path, [args])
    # All layer scripts accept the instrument name as sys.argv[1].
    # -----------------------------------------------------------------------
    common_stages = [
        ("Bronze Layer",  LAYER_SCRIPTS["bronze"],  [instrument]),
        ("Silver Layer",  LAYER_SCRIPTS["silver"],  [instrument]),
        ("Gold Layer",    LAYER_SCRIPTS["gold"],    [instrument]),
    ]

    if pipeline == "dt":
        tail_stages = [
            ("Platinum Data Prepper",        LAYER_SCRIPTS["pt_prepper"],  [instrument]),
            ("Platinum Strategy Discoverer", LAYER_SCRIPTS["pt_discover"], [instrument]),
        ]
        finish_msg = (
            f"Decision Tree pipeline complete for {instrument}!\n"
            f"{colors.YELLOW}Strategies saved in data/platinum/discovered_strategies/{colors.ENDC}"
        )
    else:
        tail_stages = [
            ("Platinum Dataset Builder", LAYER_SCRIPTS["pt_builder"], [instrument]),
            ("Diamond Trainer",          LAYER_SCRIPTS["diamond"],    [instrument]),
        ]
        finish_msg = (
            f"XGBoost pipeline complete for {instrument}!\n"
            f"{colors.YELLOW}Model saved in data/diamond/strategies/{colors.ENDC}"
        )

    all_stages = common_stages + tail_stages

    # -----------------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------------
    print(f"\n{colors.BOLD}{colors.BLUE}--- Running {len(all_stages)}-stage pipeline for: "
          f"{instrument} ---{colors.ENDC}")

    for label, script_path, args in all_stages:
        sleep(1)
        if not run_layer(label, script_path, args=args):
            print(f"\n{colors.RED}{colors.BOLD}PIPELINE HALTED — error in: {label}{colors.ENDC}")
            print(f"{colors.YELLOW}Check log files and the output above for details.{colors.ENDC}")
            return

    print(f"\n{colors.GREEN}{colors.BOLD}>>>>> {finish_msg} <<<<<{colors.ENDC}")


if __name__ == "__main__":
    main()