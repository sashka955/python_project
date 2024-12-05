from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Union
import numpy as np

@dataclass
class Descriptor:
    """Class for summarizing and describing real estate data."""
    data: List[Dict[str, Any]]

    def _validate_columns(self, columns: List[str]) -> List[str]:
        """Helper method to validate and extract the specified columns."""
        if columns == "all":
            return list(self.data[0].keys()) if self.data else []
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' not found in the data.")
        return columns

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        """
        Compute the ratio of None values per column.
        """
        columns = self._validate_columns(columns)
        result = {}
        for column in columns:
            none_count = sum(1 for row in self.data if row[column] is None)
            total_count = len(self.data)
            result[column] = none_count / total_count if total_count > 0 else 0
        return result

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        """
        Compute the average value for numeric columns.
        """
        columns = self._validate_columns(columns)
        result = {}
        for column in columns:
            values = [row[column] for row in self.data if isinstance(row[column], (int, float)) and row[column] is not None]
            if not values:
                raise ValueError(f"No numeric values in column '{column}'.")
            result[column] = sum(values) / len(values)
        return result

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        """
        Compute the median value for numeric columns.
        """
        columns = self._validate_columns(columns)
        result = {}
        for column in columns:
            values = sorted(row[column] for row in self.data if isinstance(row[column], (int, float)) and row[column] is not None)
            if not values:
                raise ValueError(f"No numeric values in column '{column}'.")
            mid = len(values) // 2
            if len(values) % 2 == 0:  # Even number of elements
                result[column] = (values[mid - 1] + values[mid]) / 2
            else:  # Odd number of elements
                result[column] = values[mid]
        return result

    def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
        """
        Compute a specific percentile value for numeric columns.
        """
        columns = self._validate_columns(columns)
        result = {}
        for column in columns:
            values = sorted(row[column] for row in self.data if isinstance(row[column], (int, float)) and row[column] is not None)
            if not values:
                raise ValueError(f"No numeric values in column '{column}'.")
            k = (len(values) - 1) * (percentile / 100)
            f = int(k)
            c = f + 1
            if c < len(values):
                result[column] = values[f] + (k - f) * (values[c] - values[f])
            else:
                result[column] = values[f]
        return result

    def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
        """
        Compute the mode (most frequent value) for each column.
        """
        columns = self._validate_columns(columns)
        result = {}
        for column in columns:
            values = [row[column] for row in self.data if row[column] is not None]
            if not values:
                raise ValueError(f"No values in column '{column}'.")
            most_common = max(set(values), key=values.count)
            if all(isinstance(value, (int, float)) for value in values):
                result[column] = ("numeric", most_common)
            elif all(isinstance(value, str) for value in values):
                result[column] = ("categorical", most_common)
            else:
                raise ValueError(f"Mixed types in column '{column}'.")
        return result

@dataclass
class DescriptorNumpy:
    """Class for summarizing and describing real estate data using NumPy."""
    data: List[Dict[str, Union[int, float, str, None]]]

    def _convert_to_numpy(self, columns: List[str]) -> Dict[str, np.ndarray]:
        """Helper method to convert columns to NumPy arrays."""
        arrays = {}
        for column in columns:
            values = [
                row.get(column, None) for row in self.data
            ]  # Extract column values, defaulting to None
            if all(isinstance(v, (int, float, type(None))) for v in values):  # Numeric
                arrays[column] = np.array(
                    [v if v is not None else np.nan for v in values], dtype=np.float64
                )
            else:  # Categorical
                arrays[column] = np.array(values, dtype=object)
        return arrays

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the ratio of None (or np.nan) values per column."""
        if columns == "all":
            columns = list(self.data[0].keys())
        arrays = self._convert_to_numpy(columns)
        return {
            column: np.isnan(arrays[column]).sum() / len(arrays[column])
            for column in arrays
        }

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric columns."""
        if columns == "all":
            columns = list(self.data[0].keys())
        arrays = self._convert_to_numpy(columns)
        result = {}
        for column, array in arrays.items():
            if array.dtype == np.float64:  # Only process numeric arrays
                result[column] = np.nanmean(array)
        return result

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric columns."""
        if columns == "all":
            columns = list(self.data[0].keys())
        arrays = self._convert_to_numpy(columns)
        result = {}
        for column, array in arrays.items():
            if array.dtype == np.float64:  # Only process numeric arrays
                result[column] = np.nanmedian(array)
        return result

    def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute a specific percentile value for numeric columns."""
        if columns == "all":
            columns = list(self.data[0].keys())
        arrays = self._convert_to_numpy(columns)
        result = {}
        for column, array in arrays.items():
            if array.dtype == np.float64:  # Only process numeric arrays
                result[column] = np.nanpercentile(array, percentile)
        return result

    def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Tuple[str, Union[float, str]]]:
        """Compute the mode for each column, determining if it is numeric or categorical."""
        if columns == "all":
            columns = list(self.data[0].keys())
        arrays = self._convert_to_numpy(columns)
        result = {}
        for column, array in arrays.items():
            if array.dtype == np.float64:  # Numeric
                non_nan_values = array[~np.isnan(array)]
                if len(non_nan_values) > 0:
                    mode = np.bincount(non_nan_values.astype(int)).argmax()
                    result[column] = ("numeric", mode)
                else:
                    result[column] = ("numeric", None)
            else:  # Categorical
                unique, counts = np.unique(array, return_counts=True)
                mode = unique[np.argmax(counts)]
                result[column] = ("categorical", mode)
        return result
