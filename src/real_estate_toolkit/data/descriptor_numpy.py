from dataclasses import dataclass
from typing import List, Dict, Union, Tuple
import numpy as np

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
