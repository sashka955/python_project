import sys
from pathlib import Path
# Add the parent directory of `src` to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List
from real_estate_toolkit.agent_based_model.house import House
from real_estate_toolkit.agent_based_model.house_market import HousingMarket


class Segment(Enum):
    FANCY = auto()  # Prefers new construction with high quality scores
    OPTIMIZER = auto()  # Focuses on price per square foot value
    AVERAGE = auto()  # Considers average market prices


@dataclass
class Consumer:
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over time using compound interest.

        Args:
            years (int): Number of years to calculate savings for.
        """
        for _ in range(years):
            annual_savings = self.annual_income * self.saving_rate
            self.savings += annual_savings
            self.savings += self.savings * self.interest_rate

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to purchase a suitable house based on the consumer's segment.

        Args:
            housing_market (HousingMarket): The housing market to search in.
        """
        if self.savings == 0:
            print(f"Consumer {self.id} has no savings to purchase a house.")
            return

        # Get all available houses in the market
        available_houses = [house for house in housing_market.houses if house.available]

        if not available_houses:
            print(f"No available houses for Consumer {self.id}.")
            return

        chosen_house = None

        if self.segment == Segment.FANCY:
            # Select new construction with the highest quality score
            high_quality_houses = [
                house for house in available_houses
                if house.is_new_construction() and house.quality_score and house.quality_score.value >= 4
            ]
            if high_quality_houses:
                chosen_house = max(high_quality_houses, key=lambda h: h.quality_score.value)

        elif self.segment == Segment.OPTIMIZER:
            # Select house with the best price per square foot
            affordable_houses = [house for house in available_houses if house.price <= self.savings]
            if affordable_houses:
                chosen_house = min(affordable_houses, key=lambda h: h.calculate_price_per_square_foot())

        elif self.segment == Segment.AVERAGE:
            # Select house priced below market average
            average_price = housing_market.calculate_average_price()
            below_average_houses = [house for house in available_houses if house.price <= average_price]
            if below_average_houses:
                chosen_house = min(below_average_houses, key=lambda h: h.price)

        # Attempt to purchase the chosen house
        if chosen_house and self.savings >= chosen_house.price:
            self.house = chosen_house
            self.savings -= chosen_house.price
            chosen_house.sell_house()
            print(f"Consumer {self.id} purchased house {chosen_house.id}.")
        else:
            print(f"Consumer {self.id} could not afford or find a suitable house.")


