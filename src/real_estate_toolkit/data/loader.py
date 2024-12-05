from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import csv


@dataclass
class DataLoader:
    """Class for loading and basic processing of real estate data."""
    data_path: Path
     

    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """
        Load data from CSV file into a list of dictionaries.
    
        Returns:
            List[Dict[str, Any]]: List of dictionaries with the data.
        """
        required_columns = ["id", "sale_price", "lot_area", "year_built", "bedroom_abv_gr"]
        try:
            with self.data_path.open("r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                fieldnames = [col.lower().replace(' ', '_') for col in reader.fieldnames]
                missing_columns = set(required_columns) - set(fieldnames)
                if missing_columns:
                    raise ValueError(f"Required columns missing from dataset: {missing_columns}")
                data = []
                for row in reader:
                    updated_row = {fieldnames[i]: value for i, (key, value) in enumerate(row.items())}
                    data.append(updated_row)
                return data
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    

    def validate_columns(self, required_columns: List[str]) -> bool:
        """
        Validate that all required columns are present in the dataset.

        Args:
            required_columns (List[str]): List of required column names.

        Returns:
            bool: True if all required columns are present, False otherwise.
        """
        try:
            with self.data_path.open("r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    raise ValueError("The CSV file has no headers.")
                # Convert field names in the file to snake_case for consistency
                fieldnames = [col.lower().replace(' ', '_') for col in reader.fieldnames]
                # Convert required columns to snake_case
                required_columns_snake_case = [col.lower().replace(' ', '_') for col in required_columns]
                # Check if all required columns are present in the field names of the CSV
                return all(column in fieldnames for column in required_columns_snake_case)
        except Exception as e:
            raise ValueError(f"Error validating columns in {self.data_path}: {e}")
