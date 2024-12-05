from enum import Enum
from dataclasses import dataclass
from typing import Optional


class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1


@dataclass
class House:
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore] = None
    available: bool = True

    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.
        Returns:
            float: The price per square foot, rounded to 2 decimal places.
        """
        if self.area <= 0:  # Handle edge case for area = 0 or negative
            return 0.0
        return round(self.price / self.area, 2)

    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if the house is considered new construction (< 5 years old).
        Args:
            current_year (int): The current year. Defaults to 2024.
        Returns:
            bool: True if the house is new construction, False otherwise.
        """
        return (current_year - self.year_built) < 5

    def get_quality_score(self) -> None:
        """
        Generate a quality score based on house attributes.
        If the quality_score is not already set, calculate it based on:
        - Age of the house
        - Size (area)
        - Number of bedrooms
        """
        if self.quality_score is not None:
            return  # If quality_score is already set, do nothing

        # Basic scoring logic
        age = 2024 - self.year_built
        score = 5  # Start with EXCELLENT score
        
        # Deduct points based on age
        if age > 50:
            score -= 2
        elif age > 20:
            score -= 1

        # Bonus for size and bedrooms
        if self.area >= 2000:
            score += 1
        if self.bedrooms >= 4:
            score += 1

        # Ensure score stays within bounds
        score = max(1, min(score, 5))

        # Assign quality score
        self.quality_score = QualityScore(score)

    def sell_house(self) -> None:
        """
        Mark house as sold.
        Updates:
            - Sets available status to False.
        """
        self.available = False
