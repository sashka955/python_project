import sys
from pathlib import Path
# Add the parent directory of `src` to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from enum import Enum, auto
from dataclasses import dataclass, field
from random import gauss, randint, shuffle
from typing import Any, List, Dict
from real_estate_toolkit.agent_based_model.house import House
from real_estate_toolkit.agent_based_model.house_market import HousingMarket
from real_estate_toolkit.agent_based_model.consumer import Consumer, Segment


class CleaningMarketMechanism(Enum):
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()


@dataclass
class AnnualIncomeStatistics:
    minimum: float
    average: float
    standard_deviation: float
    maximum: float


@dataclass
class ChildrenRange:
    minimum: int = 0
    maximum: int = 5


@dataclass
class Simulation:
    housing_market_data: List[Dict[str, Any]]
    consumers_number: int
    years: int
    annual_income: AnnualIncomeStatistics
    children_range: ChildrenRange
    cleaning_market_mechanism: CleaningMarketMechanism
    down_payment_percentage: float = 0.2
    saving_rate: float = 0.3
    interest_rate: float = 0.05
    housing_market: HousingMarket = field(init=False)
    consumers: List[Consumer] = field(init=False)

    def create_housing_market(self):
        """
        Initialize market with houses.
        """
        houses = [
            House(**house_data)
            for house_data in self.housing_market_data
        ]
        self.housing_market = HousingMarket(houses=houses)

    def create_consumers(self) -> None:
        """
        Generate a population of consumers.
        """
        consumers = []
        for _ in range(self.consumers_number):
            # Generate annual income
            while True:
                income = gauss(self.annual_income.average, self.annual_income.standard_deviation)
                if self.annual_income.minimum <= income <= self.annual_income.maximum:
                    break

            # Generate children number
            children_number = randint(self.children_range.minimum, self.children_range.maximum)

            # Assign a random segment
            segment = Segment(randint(1, len(Segment)))

            # Create a consumer
            consumer = Consumer(
                id=len(consumers) + 1,
                annual_income=income,
                children_number=children_number,
                segment=segment,
                savings=0.0,
                saving_rate=self.saving_rate,
                interest_rate=self.interest_rate,
            )
            consumers.append(consumer)

        self.consumers = consumers

    def compute_consumers_savings(self) -> None:
        """
        Calculate savings for all consumers over the simulation period.
        """
        for consumer in self.consumers:
            consumer.compute_savings(self.years)

    def clean_the_market(self) -> None:
        """
        Execute market transactions using the specified cleaning mechanism.
        """
        if self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income, reverse=True)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.RANDOM:
            shuffle(self.consumers)

        for consumer in self.consumers:
            consumer.buy_a_house(self.housing_market)

    def compute_owners_population_rate(self) -> float:
        """
        Compute the percentage of consumers who bought a house.
        """
        owners = sum(1 for consumer in self.consumers if consumer.house is not None)
        return owners / len(self.consumers)

    def compute_houses_availability_rate(self) -> float:
        """
        Compute the percentage of houses still available in the market.
        """
        available_houses = sum(1 for house in self.housing_market.houses if house.available)
        return available_houses / len(self.housing_market.houses)
