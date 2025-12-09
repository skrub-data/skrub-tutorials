"""
Script to generate synthetic dataframes with numeric and categorical features.

This script allows you to create a Polars DataFrame with:
- Configurable number of rows and columns
- Both numeric and categorical (text) features
- Columns with specified fractions of null values
- Columns with single constant values
- Datetime columns with custom formats
- Add missing values to existing columns
"""

import polars as pl
import random
from typing import Optional, List, Dict
from datetime import datetime, timedelta


# Sample data for categorical features
FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
    "Lucas", "Harper", "Henry", "Evelyn", "Alexander", "Abigail", "Michael",
    "Emily", "Daniel", "Elizabeth", "Matthew", "Sofia", "Jackson", "Avery",
    "Sebastian", "Ella", "Jack", "Scarlett", "Aiden", "Grace", "Owen", "Chloe",
    "Samuel", "Victoria", "David", "Riley", "Joseph", "Aria", "Carter", "Lily"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green"
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis",
    "Seattle", "Denver", "Washington", "Boston", "Nashville", "Detroit",
    "Portland", "Las Vegas", "Memphis", "Louisville", "Baltimore", "Milwaukee",
    "Albuquerque", "Tucson", "Fresno", "Mesa", "Sacramento", "Atlanta",
    "Kansas City", "Colorado Springs", "Raleigh", "Miami", "Long Beach"
]

COUNTRIES = [
    "USA", "Canada", "Mexico", "UK", "France", "Germany", "Italy", "Spain",
    "Netherlands", "Belgium", "Switzerland", "Austria", "Sweden", "Norway",
    "Denmark", "Finland", "Poland", "Czech Republic", "Portugal", "Ireland",
    "Australia", "New Zealand", "Japan", "South Korea", "Singapore", "Brazil",
    "Argentina", "Chile", "Colombia", "Peru", "India", "China", "Thailand"
]

DEPARTMENTS = [
    "Engineering", "Sales", "Marketing", "Human Resources", "Finance",
    "Operations", "Customer Support", "Product", "Research", "Legal",
    "IT", "Analytics", "Design", "Quality Assurance", "Procurement"
]

PRODUCTS = [
    "Laptop", "Desktop", "Tablet", "Smartphone", "Monitor", "Keyboard",
    "Mouse", "Headphones", "Webcam", "Printer", "Scanner", "Router",
    "Hard Drive", "SSD", "RAM", "Graphics Card", "Motherboard", "CPU"
]


def generate_synthetic_dataframe(
    n_rows: int,
    n_numeric: int = 3,
    n_categorical: int = 3,
    n_null_columns: int = 0,
    null_fraction: float = 0.3,
    n_constant_columns: int = 0,
    constant_column_name: str = "constant",
    constant_value: str = "CONSTANT",
    n_datetime_columns: int = 0,
    datetime_format: str = "%Y-%m-%d",
    columns_with_nulls: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None
) -> pl.DataFrame:
    """
    Generate a synthetic Polars DataFrame with numeric and categorical features.
    
    Parameters
    ----------
    n_rows : int
        Number of rows in the DataFrame
    n_numeric : int, default=3
        Number of numeric columns to generate
    n_categorical : int, default=3
        Number of categorical (text) columns to generate
    n_null_columns : int, default=0
        Number of additional columns with null values
    null_fraction : float, default=0.3
        Fraction of null values in null columns (0.0 to 1.0)
    n_constant_columns : int, default=0
        Number of columns with a single constant value
    constant_column_name : str, default="constant"
        The name prefix for constant columns
    constant_value : str, default="CONSTANT"
        The constant value to use for constant columns
    n_datetime_columns : int, default=0
        Number of datetime columns to generate
    datetime_format : str, default="%Y-%m-%d"
        Format string for datetime columns (e.g., "%d-%b-%Y" for "01-Jan-2024")
    columns_with_nulls : dict, optional
        Dictionary mapping existing column names to null fractions.
        For example: {"first_name": 0.1, "city": 0.2}
        This adds missing values to existing columns after generation.
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pl.DataFrame
        Generated synthetic DataFrame
        
    Examples
    --------
    >>> # Basic usage
    >>> df = generate_synthetic_dataframe(n_rows=100)
    >>> 
    >>> # Custom configuration with datetime columns
    >>> df = generate_synthetic_dataframe(
    ...     n_rows=500,
    ...     n_numeric=5,
    ...     n_categorical=4,
    ...     n_null_columns=2,
    ...     null_fraction=0.2,
    ...     n_constant_columns=1,
    ...     n_datetime_columns=2,
    ...     datetime_format="%d-%b-%Y",
    ...     columns_with_nulls={"first_name": 0.1, "city": 0.15},
    ...     seed=42
    ... )
    """
    if seed is not None:
        random.seed(seed)
    
    data = {}
    
    # Generate numeric columns
    for i in range(n_numeric):
        col_name = f"num_{i+1}"
        # Mix of different numeric distributions
        if i % 3 == 0:
            # Integer values
            data[col_name] = [random.randint(0, 100) for _ in range(n_rows)]
        elif i % 3 == 1:
            # Float values
            data[col_name] = [random.uniform(0, 1000) for _ in range(n_rows)]
        else:
            # Normally distributed values
            data[col_name] = [random.gauss(50, 15) for _ in range(n_rows)]
    
    # Generate categorical columns
    categorical_sources = [
        ("first_name", FIRST_NAMES),
        ("last_name", LAST_NAMES),
        ("city", CITIES),
        ("country", COUNTRIES),
        ("department", DEPARTMENTS),
        ("product", PRODUCTS),
    ]
    
    for i in range(n_categorical):
        if i < len(categorical_sources):
            col_name, source_list = categorical_sources[i]
            col_name = f"{col_name}"
        else:
            # Use a cycling pattern for additional categorical columns
            source_idx = i % len(categorical_sources)
            col_name, source_list = categorical_sources[source_idx]
            col_name = f"{col_name}_{i+1}"
        
        data[col_name] = [random.choice(source_list) for _ in range(n_rows)]
    
    # Generate columns with null values
    for i in range(n_null_columns):
        col_name = f"with_nulls_{i+1}"
        values = []
        for _ in range(n_rows):
            if random.random() < null_fraction:
                values.append(None)
            else:
                values.append(random.choice(CITIES))
        data[col_name] = values
    
    # Generate constant columns
    for i in range(n_constant_columns):
        col_name = f"{constant_column_name}_{i+1}"
        data[col_name] = [constant_value] * n_rows
    
    # Generate datetime columns
    base_date = datetime(2020, 1, 1)
    for i in range(n_datetime_columns):
        col_name = f"date_{i+1}"
        dates = []
        for _ in range(n_rows):
            days_offset = random.randint(0, 1461)  # ~4 years
            date = base_date + timedelta(days=days_offset)
            # Format the date as a string using the provided format
            dates.append(date.strftime(datetime_format))
        data[col_name] = dates
    
    # Convert to DataFrame
    df = pl.DataFrame(data)
    
    # Add nulls to existing columns if specified
    if columns_with_nulls:
        for col_name, null_frac in columns_with_nulls.items():
            if col_name in df.columns:
                # Create a mask of null positions
                mask = [random.random() < null_frac for _ in range(n_rows)]
                # Replace values with None where mask is True
                values = df[col_name].to_list()
                values = [None if m else v for m, v in zip(mask, values)]
                df = df.with_columns(pl.Series(col_name, values))
    
    return df


def main():
    """Example usage of the synthetic data generator."""
    print("Generating synthetic DataFrame...\n")
    
    df = generate_synthetic_dataframe(
        n_rows=10000,
        n_numeric=3,
        n_categorical=5,
        n_null_columns=1,
        null_fraction=0.75,
        n_constant_columns=1,
        constant_column_name="contract_type",
        constant_value="CONTRACT",
        n_datetime_columns=2,
        datetime_format="%d-%b-%Y",
        columns_with_nulls={"first_name": 0.1, "city": 0.15},
        seed=123
    )
    
    print(f"Large DataFrame shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())
    print("Generated DataFrame:")
    print(df)
    print(f"\nShape: {df.shape}")
    print("\nColumn types:")
    print(df.schema)
    print("\nNull counts:")
    print(df.null_count())
    
    df.write_csv("synthetic_data.csv")

if __name__ == "__main__":
    main()
