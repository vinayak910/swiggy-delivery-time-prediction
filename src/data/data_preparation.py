import pandas as pd 
import numpy as np 
import yaml
import logging 
from pathlib import Path 
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


# Target column name
TARGET = "time_taken"


# -------------------- LOGGER SETUP --------------------
logger = logging.getLogger(name="data_preparation")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Prevent adding duplicate handlers if script runs multiple times
if not logger.hasHandlers():
    logger.addHandler(handler)


# -------------------- FUNCTION DEFINITIONS --------------------

def read_params(file_path: Path) -> dict:
    """
    Reads the parameters from a YAML config file.

    Args:
        file_path (Path): Path to the YAML file.

    Returns:
        dict: Dictionary of parameters.
    """
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Params loaded successfully from {file_path}")
        return params
    except Exception as e:
        logger.error(f"Error reading params file: {e}")
        raise


def load_data(data_path: Path) -> Optional[pd.DataFrame]:
    """
    Loads data from a CSV file.

    Args:
        data_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if file not found.
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {data_path}")
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
    return None


def train_test_split_data(df: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train and test subsets.

    Args:
        df (pd.DataFrame): Input DataFrame.
        test_size (float): Proportion of test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and Test DataFrames.
    """
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logger.info(f"Data split into train ({1 - test_size:.0%}) and test ({test_size:.0%})")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error during train-test split: {e}")
        raise


def save_data(df: pd.DataFrame, save_path: Path) -> None:
    """
    Saves the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Data to save.
        save_path (Path): Destination file path.

    Returns:
        None
    """
    try:
        df.to_csv(save_path, index=False)
        logger.info(f"Data saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving data to {save_path}: {e}")
        raise


# -------------------- MAIN EXECUTION --------------------

if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"
    save_data_dir = root_path / "data" / "interim"
    save_data_dir.mkdir(parents=True, exist_ok=True)

    train_file_name = "train.csv"
    test_file_name = "test.csv"
    save_train_path = save_data_dir / train_file_name
    save_test_path = save_data_dir / test_file_name

    params_file_path = root_path / "params.yaml"

    # Load data
    df = load_data(data_path=data_path)
    if df is None:
        logger.error("Data loading failed. Exiting script.")
        exit(1)

    # Load parameters
    params = read_params(params_file_path)
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)

    # Split data
    train_data, test_data = train_test_split_data(df, test_size=test_size, random_state=random_state)

    # Save datasets
    data_subsets = [train_data, test_data]
    data_paths = [save_train_path, save_test_path]
    file_name_list = [train_file_name, test_file_name]

    for filename, path, data in zip(file_name_list, data_paths, data_subsets):
        save_data(df=data, save_path=path)
