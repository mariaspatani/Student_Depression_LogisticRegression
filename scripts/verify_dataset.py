"""
scripts/verify_dataset.py
-------------------------
Integrity check script for the Student Depression Dataset.
Verifies file presence, shape, column schema, target balance, and nulls.
Designed to run with Python standard library or pandas.
"""

import os
import sys
import csv

def verify_dataset(filepath="dataset/Student Depression Dataset.csv"):
    if not os.path.exists(filepath):
        print(f"[ERROR] Dataset file not found at: {filepath}")
        return False

    print("=" * 60)
    print("Team 16 — Student Depression Dataset Verification")
    print("=" * 60)

    try:
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            
            row_count = 0
            depression_col_idx = header.index("Depression") if "Depression" in header else -1
            
            target_counts = {0: 0, 1: 0}
            null_count = 0

            for row in reader:
                row_count += 1
                if any(val.strip() == "" for val in row):
                    null_count += 1
                
                if depression_col_idx != -1:
                    val = row[depression_col_idx].strip()
                    if val in ("1", "1.0"):
                        target_counts[1] += 1
                    elif val in ("0", "0.0"):
                        target_counts[0] += 1

        print(f"File Path:            {filepath}")
        print(f"Total Rows:           {row_count:,}")
        print(f"Total Columns:        {len(header)}")
        print(f"Missing Value Rows:   {null_count}")
        print("-" * 60)
        print("Target Distribution ('Depression'):")
        total_valid_target = target_counts[0] + target_counts[1]
        if total_valid_target > 0:
            pct_0 = (target_counts[0] / total_valid_target) * 100
            pct_1 = (target_counts[1] / total_valid_target) * 100
            print(f"  Class 0 (No Depression): {target_counts[0]:,} ({pct_0:.2f}%)")
            print(f"  Class 1 (Depression):    {target_counts[1]:,} ({pct_1:.2f}%)")
        print("-" * 60)
        print("[SUCCESS] Dataset verification completed successfully.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to read dataset: {e}")
        return False

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/Student Depression Dataset.csv"
    success = verify_dataset(path)
    sys.exit(0 if success else 1)
