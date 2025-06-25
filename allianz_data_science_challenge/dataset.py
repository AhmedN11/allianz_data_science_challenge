import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging
from allianz_data_science_challenge.config import RAW_DATA_DIR


def load_data():
    """
    Load the raw dataset from the data directory.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Try common data file names and formats
    file_path = RAW_DATA_DIR / "bank-additional-full.csv"
    if file_path.exists():
        logging.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path, sep=';')
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    # If no file found, raise an error with helpful message
    raise FileNotFoundError(
        f"No data file found. Please place your dataset in this location: {file_path}"
    )