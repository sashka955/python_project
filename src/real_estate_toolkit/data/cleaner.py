from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Cleaner:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        """
        Rename the columns with best practices (e.g., snake_case, descriptive names).
        Modifies the data in place.
        """
        if not self.data:
            return  # If no data, do nothing

        # Get original keys from the first dictionary (column names)
        original_keys = list(self.data[0].keys())

        # Generate new keys in snake_case manually
        new_keys = {}
        for key in original_keys:
            new_key = key.strip().lower().replace(" ", "_").replace("-", "_")
            new_keys[key] = new_key

        # Apply renaming to all rows
        for row in self.data:
            for old_key, new_key in new_keys.items():
                row[new_key] = row.pop(old_key)

    def na_to_none(self) -> List[Dict[str, Any]]:
        """
        Replace 'NA' with None in all values in the dataset.
        
        Returns:
            List[Dict[str, Any]]: A modified list of dictionaries with 'NA' replaced by None.
        """
        return [
            {key: (None if value == "NA" else value) for key, value in row.items()}
            for row in self.data
        ]
