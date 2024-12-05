from typing import List, Dict
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import os


class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.
        
        Args:
            data_path (str): Path to the Ames Housing dataset
        """
        self.data_path = data_path
        self.real_estate_data = pl.read_csv(data_path)
        self.real_estate_clean_data = None
        self.output_folder = "src/real_estate_toolkit/analytics/outputs"
        os.makedirs(self.output_folder, exist_ok=True)

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning.
        """
        # Handle missing values
        self.real_estate_clean_data = self.real_estate_data.fill_null(strategy="mean")

        # Ensure proper data types
        numeric_columns = [col for col in self.real_estate_clean_data.columns if self.real_estate_clean_data[col].dtype in [pl.Float64, pl.Int64]]
        categorical_columns = [col for col in self.real_estate_clean_data.columns if self.real_estate_clean_data[col].dtype == pl.Utf8]
        
        for col in numeric_columns:
            self.real_estate_clean_data = self.real_estate_clean_data.with_column(pl.col(col).cast(pl.Float64))
        
        for col in categorical_columns:
            self.real_estate_clean_data = self.real_estate_clean_data.with_column(pl.col(col).cast(pl.Categorical))

    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before analysis.")

        # Compute statistics
        price_statistics = self.real_estate_clean_data["SalePrice"].describe()

        # Create histogram
        fig = px.histogram(
            self.real_estate_clean_data.to_pandas(),
            x="SalePrice",
            nbins=30,
            title="Sale Price Distribution",
            labels={"SalePrice": "Sale Price"},
            template="plotly_white",
        )
        fig.update_layout(xaxis_title="Sale Price", yaxis_title="Count")
        fig_path = os.path.join(self.output_folder, "price_distribution.html")
        fig.write_html(fig_path)

        return price_statistics

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across neighborhoods.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before analysis.")

        # Group data by neighborhood and compute statistics
        neighborhood_stats = self.real_estate_clean_data.groupby("Neighborhood").agg([
            pl.col("SalePrice").mean().alias("MeanPrice"),
            pl.col("SalePrice").median().alias("MedianPrice"),
            pl.col("SalePrice").std().alias("PriceStd"),
            pl.col("SalePrice").min().alias("MinPrice"),
            pl.col("SalePrice").max().alias("MaxPrice"),
        ])

        # Create boxplot
        fig = px.box(
            self.real_estate_clean_data.to_pandas(),
            x="Neighborhood",
            y="SalePrice",
            title="Neighborhood Price Comparison",
            labels={"SalePrice": "Sale Price", "Neighborhood": "Neighborhood"},
            template="plotly_white",
        )
        fig.update_layout(xaxis_title="Neighborhood", yaxis_title="Sale Price")
        fig_path = os.path.join(self.output_folder, "neighborhood_price_comparison.html")
        fig.write_html(fig_path)

        return neighborhood_stats

    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for specified variables.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before analysis.")

        # Compute correlation matrix
        correlation_matrix = self.real_estate_clean_data.select(variables).to_pandas().corr()

        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            labels={"color": "Correlation"},
            x=variables,
            y=variables,
            color_continuous_scale="viridis",
        )
        fig_path = os.path.join(self.output_folder, "feature_correlation_heatmap.html")
        fig.write_html(fig_path)

    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """
        Create scatter plots exploring relationships between key features.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before analysis.")

        scatter_plots = {}

        # Scatter plot 1: Sale Price vs Total Square Footage
        fig1 = px.scatter(
            self.real_estate_clean_data.to_pandas(),
            x="GrLivArea",
            y="SalePrice",
            title="Sale Price vs Total Square Footage",
            labels={"GrLivArea": "Total Square Footage", "SalePrice": "Sale Price"},
            trendline="ols",
            color="Neighborhood",
            template="plotly_white",
        )
        fig1_path = os.path.join(self.output_folder, "scatter_price_vs_sqft.html")
        fig1.write_html(fig1_path)
        scatter_plots["price_vs_sqft"] = fig1

        # Scatter plot 2: Sale Price vs Year Built
        fig2 = px.scatter(
            self.real_estate_clean_data.to_pandas(),
            x="YearBuilt",
            y="SalePrice",
            title="Sale Price vs Year Built",
            labels={"YearBuilt": "Year Built", "SalePrice": "Sale Price"},
            trendline="ols",
            color="Neighborhood",
            template="plotly_white",
        )
        fig2_path = os.path.join(self.output_folder, "scatter_price_vs_year.html")
        fig2.write_html(fig2_path)
        scatter_plots["price_vs_year"] = fig2

        # Scatter plot 3: Sale Price vs Overall Quality
        fig3 = px.scatter(
            self.real_estate_clean_data.to_pandas(),
            x="OverallQual",
            y="SalePrice",
            title="Sale Price vs Overall Quality",
            labels={"OverallQual": "Overall Quality", "SalePrice": "Sale Price"},
            trendline="ols",
            color="Neighborhood",
            template="plotly_white",
        )
        fig3_path = os.path.join(self.output_folder, "scatter_price_vs_quality.html")
        fig3.write_html(fig3_path)
        scatter_plots["price_vs_quality"] = fig3

        return scatter_plots
