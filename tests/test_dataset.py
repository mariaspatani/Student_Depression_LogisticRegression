"""
tests/test_dataset.py
---------------------
Unit tests verifying the structure, presence, and integrity of the Student Depression Dataset.
Uses Python's standard unittest framework.
"""

import os
import csv
import unittest

class TestStudentDepressionDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_path = os.path.join("dataset", "Student Depression Dataset.csv")
        cls.assert_file_exists = os.path.isfile(cls.dataset_path)

    def test_file_exists(self):
        """Dataset file must exist at dataset/Student Depression Dataset.csv"""
        self.assertTrue(self.assert_file_exists, f"Missing file: {self.dataset_path}")

    def test_header_and_target_presence(self):
        """Dataset must contain target column 'Depression' and feature columns."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        
        self.assertIn("Depression", header)
        self.assertIn("Academic Pressure", header)
        self.assertIn("Financial Stress", header)
        self.assertIn("Study Satisfaction", header)
        self.assertEqual(len(header), 18)

    def test_row_count(self):
        """Dataset must contain approximately 27,901 records."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader)
            row_count = sum(1 for _ in reader)
            
        self.assertEqual(row_count, 27901)

    def test_target_values_are_binary(self):
        """Target 'Depression' values must be strictly binary (0 or 1)."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            target_idx = header.index("Depression")
            
            unique_targets = set()
            for row in reader:
                val = row[target_idx].strip()
                if val:
                    unique_targets.add(val)

        self.assertTrue(unique_targets.issubset({"0", "1", "0.0", "1.0"}))

if __name__ == "__main__":
    unittest.main()
