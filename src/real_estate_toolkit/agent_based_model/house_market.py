import sys
from pathlib import Path
# Add the parent directory of `src` to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from typing import List, Optional
from real_estate_toolkit.agent_based_model.house import House




class HousingMarket:
    def __init__(self, houses: List[House]):
        """
        Initialize the housing market with a list of houses.
        Args:
            houses (List[House]): List of House objects.
        """
        self.houses: List[House] = houses

    def get_house_by_id(self, house_id: int) -> Optional[House]:
        """
        Retrieve specific house by ID.
        
        Args:
            house_id (int): The ID of the house to retrieve.
        
        Returns:
            House: The house with the specified ID, or None if not found.
        """
        for house in self.houses:
            if house.id == house_id:
                return house
        return None  # Handle non-existent IDs

    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate the average house price, optionally filtered by bedrooms.
        
        Args:
            bedrooms (Optional[int]): Number of bedrooms to filter by.
        
        Returns:
            float: The average price of the houses, or 0.0 if no houses match.
        """
        filtered_houses = (
            [house for house in self.houses if house.bedrooms == bedrooms]
            if bedrooms is not None
            else self.houses
        )

        if not filtered_houses:  # Handle empty lists
            return 0.0

        total_price = sum(house.price for house in filtered_houses if house.price > 0)
        return total_price / len(filtered_houses)

    def get_houses_that_meet_requirements(self, max_price: int, segment: str) -> Optional[List[House]]:
        """
        Filter houses based on buyer requirements.
        
        Args:
            max_price (int): Maximum price the buyer is willing to pay.
            segment (str): Desired segment ('luxury', 'family', 'starter').
        
        Returns:
            Optional[List[House]]: List of houses that meet the requirements, or None if none match.
        """
        filtered_houses = [
            house
            for house in self.houses
            if house.price <= max_price and self._is_in_segment(house, segment)
        ]

        return filtered_houses if filtered_houses else None

    def _is_in_segment(self, house: House, segment: str) -> bool:
        """
        Determine if a house fits a specific segment.
        
        Args:
            house (House): The house to evaluate.
            segment (str): The segment ('luxury', 'family', 'starter').
        
        Returns:
            bool: True if the house fits the segment, False otherwise.
        """
        if segment == "luxury":
            return house.price > 500000 and house.quality_score and house.quality_score.value >= 4
        elif segment == "family":
            return house.bedrooms >= 3 and house.area >= 1500
        elif segment == "starter":
            return house.price < 300000 and house.bedrooms <= 2
        return False  # If the segment is not recognized
