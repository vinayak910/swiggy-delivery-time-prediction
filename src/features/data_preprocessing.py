import pandas as pd 
import numpy as np 
import joblib 
import logging
from pathlib import Path 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder 


from sklearn import set_config 

set_config(transform_output= "pandas")


logger = logging.getLogger(name = "data_preprocessing")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

traffic_order = ["low","medium","high","jam"]

distance_type_order = ["short","medium","long","very_long"]

target_col = 'time_taken'

def load_data(data_path: Path)->pd.DataFrame:
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


def drop_missing_values(df:pd.DataFrame)->pd.DataFrame:

    logger.info(f"The original dataset with missing values has {df.shape} shape.")

    df_dropped = df.dropna()

    logger.info(f"The dataset with missing values drop has a shape of {df_dropped.shape}")

    missing_values = df_dropped.isna().sum().sum()

    if missing_values>0:
        
        raise ValueError("The dataset still has missing values")
    
    return df_dropped


def make_X_and_y(data:pd.DataFrame , target_column:str):

    X = data.drop(columns=target_column)
    y = data[target_column]

    return X , y


def train_preprocessor(preprocessor , df: pd.DataFrame):
    preprocessor.fit(df)
    return preprocessor


def perform_transformations(preprocessor , df: pd.DataFrame):
    
    return preprocessor.transform(df)


def save_data(data:pd.DataFrame, save_path:Path)->None:
    data.to_csv(save_path , index = False)

def save_transformer(transformer, save_dir:Path, transformer_name:str):

    save_path = save_dir / transformer_name

    try:
        joblib.dump(value = transformer , filename= save_path)
        logger.info(f"{transformer_name} saved to location :{save_dir}")

    except Exception as e:
        logger.error(f"Error while saving {transformer_name} transformer.")



def join_X_and_y(X:pd.DataFrame , y:pd.Series):

    join_data = X.join(y , how = 'inner')

    return join_data



if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    train_data_path = root_path/ "data" / "interim" / "train.csv"
    test_data_path = root_path / "data" / "interim" / "test.csv"

    save_data_dir = root_path / "data" / "processed"

    save_data_dir.mkdir(parents=True ,exist_ok= True)

    train_processed_file_name = "train_processed.csv"
    test_processed_file_name = "test_processed.csv"

    save_train_processed_path = save_data_dir / train_processed_file_name
    save_test_processed_path = save_data_dir / test_processed_file_name



    preprocessor = ColumnTransformer(transformers=[
            ("scale", MinMaxScaler(), num_cols),
            ("nominal_encode", OneHotEncoder(drop="first",
                                            handle_unknown="ignore",
                                            sparse_output=False), nominal_cat_cols),
            ("ordinal_encode", OrdinalEncoder(categories=[traffic_order,
                                                          distance_type_order],
                                            encoded_missing_value=-999,
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1), ordinal_cat_cols)],
                                    remainder="passthrough",
                                    n_jobs=-1,
                                    force_int_remainder_cols=False,
                                    verbose_feature_names_out=False)


    train_df = drop_missing_values(load_data(data_path=train_data_path))
    logger.info("Train data loaded successfully")
    test_df = drop_missing_values(load_data(data_path=test_data_path))
    logger.info("Test data loaded successfully")


    X_train, y_train = make_X_and_y(data=train_df,target_column=target_col)
    X_test, y_test = make_X_and_y(data=test_df, target_column=target_col)
    logger.info("Data splitting completed")


    train_preprocessor(preprocessor=preprocessor, df=X_train)
    logger.info("Preprocessor is trained")


    X_train_trans =  perform_transformations(preprocessor=preprocessor, df=X_train)
    logger.info("Train data is transformed")
    X_test_trans = perform_transformations(preprocessor=preprocessor, df=X_test)
    logger.info("Test data is transformed")



    train_trans_df = join_X_and_y(X_train_trans, y_train)
    test_trans_df = join_X_and_y(X_test_trans, y_test)
    logger.info("Datasets joined")


    data_subsets = [train_trans_df, test_trans_df]
    data_paths = [save_train_processed_path,save_test_processed_path]
    filename_list = [train_processed_file_name, test_processed_file_name]


    for filename , path, data in zip(filename_list, data_paths, data_subsets):
        save_data(data=data, save_path=path)
        logger.info(f"{filename.replace(".csv","")} data saved to location")


    transformer_filename = "preprocessor.joblib"
    # directory to save transformers
    transformer_save_dir = root_path / "models"
    transformer_save_dir.mkdir(exist_ok=True)
    # save the transformer
    save_transformer(transformer=preprocessor,
                     save_dir=transformer_save_dir,
                     transformer_name=transformer_filename)
    logger.info("Preprocessor saved to location")