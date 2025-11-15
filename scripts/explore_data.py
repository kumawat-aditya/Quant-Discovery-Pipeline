"""
Interactive Large CSV/Parquet Explorer Utility
"""

import os
import pandas as pd


#--------------------------------------------------------------
#  Utility: Detect file type
#--------------------------------------------------------------
def is_parquet_file(path):
    return os.path.splitext(path)[1].lower() in ('.parquet', '.pq', '.parq')


#--------------------------------------------------------------
#  Utility: Read Headings
#--------------------------------------------------------------
def get_csv_headings(file_path):
    if is_parquet_file(file_path):
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(file_path)
            schema = pf.schema_arrow
            return list(schema.names)
        except Exception:
            try:
                df = pd.read_parquet(file_path)
                return df.columns.tolist()
            except Exception as e:
                print(f"[ERROR] Reading parquet headings failed: {e}")
                return None

    try:
        with pd.read_csv(file_path, iterator=True, chunksize=1, low_memory=True) as reader:
            return reader._engine.names
    except Exception as e:
        print(f"[ERROR] Reading CSV headings failed: {e}")
        return None


#--------------------------------------------------------------
#  Utility: Preview File
#--------------------------------------------------------------
def show_csv_preview(file_path, num_rows=5):
    try:
        if is_parquet_file(file_path):
            try:
                df = pd.read_parquet(file_path).head(num_rows)
            except Exception:
                import pyarrow.parquet as pq
                table = pq.read_table(file_path)
                df = table.slice(0, num_rows).to_pandas()
        else:
            df = pd.read_csv(file_path, nrows=num_rows, low_memory=True)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.expand_frame_repr", False)

        print(f"\n[SUCCESS] Showing first {num_rows} rows:\n")
        print(df.to_string(index=False))

    except Exception as e:
        print(f"[ERROR] Could not preview rows: {e}")


#--------------------------------------------------------------
#  Interactive Directory Navigator (starting at parent directory)
#--------------------------------------------------------------
def interactive_file_browser(start_path):
    current = os.path.abspath(start_path)

    while True:
        print("\n📁 Current Directory:", current)

        try:
            items = os.listdir(current)
        except PermissionError:
            print("⚠️  Permission denied. Going back.")
            current = os.path.dirname(current)
            continue

        dirs = sorted([d for d in items if os.path.isdir(os.path.join(current, d))])
        files = sorted([f for f in items
                        if os.path.isfile(os.path.join(current, f))
                        and (f.lower().endswith(".csv") or f.lower().endswith(".parquet"))])

        # Menu
        print("\nSelect an item:")
        print("  ..  ➝  Go to parent directory")
        print("  q   ➝  Quit\n")

        for i, d in enumerate(dirs):
            print(f"{i}: 🗂️  {d}/")

        for j, f in enumerate(files):
            print(f"{len(dirs)+j}: 📄 {f}")

        choice = input("\nEnter index: ").strip()

        if choice == "q":
            return None

        if choice == "..":
            parent = os.path.dirname(current)
            if parent != current:
                current = parent
            continue

        if not choice.isdigit():
            print("❌ Enter a valid number.")
            continue

        index = int(choice)

        if index < len(dirs):
            current = os.path.join(current, dirs[index])
            continue

        file_index = index - len(dirs)
        if 0 <= file_index < len(files):
            return os.path.join(current, files[file_index])

        print("❌ Invalid index.")


#--------------------------------------------------------------
#  MAIN PROGRAM
#--------------------------------------------------------------
def main():
    print("\n📁 Interactive Large CSV/Parquet Explorer")

    # ----------------------------
    # NEW: Start from project root
    # ----------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))  # ← go one directory up

    file_path = interactive_file_browser(project_root)

    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    print(f"\n📌 Selected File: {file_path}")

    print("\nWhat do you want to do?")
    print("1️⃣  Show only column headings")
    print("2️⃣  Show headings + preview rows")

    choice = input("Enter choice: ").strip()

    headings = get_csv_headings(file_path)
    if headings:
        print("\n📌 Column Headings:")
        print(headings)

        if choice == "2":
            try:
                num_rows = int(input("How many rows? (default 5): ").strip() or 5)
            except:
                num_rows = 5

            show_csv_preview(file_path, num_rows)
    else:
        print("[WARNING] Could not read headings.")


if __name__ == "__main__":
    main()
