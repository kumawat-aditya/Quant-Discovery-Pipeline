import os
from typing import List, Optional


def scan_new_files(
    input_dir: str,
    output_dir: str,
    input_ext: str = ".parquet",
    output_ext: str = ".parquet"
) -> List[str]:
    """
    Returns files present in input_dir that do not yet exist in output_dir.
    """
    input_files = sorted(f for f in os.listdir(input_dir) if f.endswith(input_ext))
    output_bases = {
        os.path.splitext(f)[0]
        for f in os.listdir(output_dir)
        if f.endswith(output_ext)
    }

    return [f for f in input_files if os.path.splitext(f)[0] not in output_bases]


def select_files_interactively(files: List[str]) -> List[str]:
    """
    Interactive selection wrapper.
    """
    if not files:
        return []

    print("\n--- Select File(s) to Process ---")
    for i, f in enumerate(files):
        print(f"  [{i+1}] {f}")
    print("  [a] Process All")
    print("\nEnter selection (e.g., 1,3 or a):")

    user_input = input("> ").strip().lower()

    if not user_input:
        return []

    if user_input == "a":
        return files

    try:
        indices = {int(i.strip()) - 1 for i in user_input.split(",")}
        return [files[i] for i in sorted(indices) if 0 <= i < len(files)]
    except ValueError:
        print("Invalid input.")
        return []
