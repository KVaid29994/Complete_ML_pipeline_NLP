import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle
import os
import yaml
from dvclive import Live

log_dr = "logs"
os.makedirs(log_dr, exist_ok= True)

logger= logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dr,"model_building.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path} with shape {df.shape}")
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the pdf file")
        raise
    except FileNotFoundError as e:
        logger.error("no file found",e)
    except Exception as e:
        logger.error("not able to load the file",e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params:dict) -> RandomForestClassifier :
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of sample in X_train and y_train must be same")
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')['model_building']
        # params = {'n_estimators' :24, 'random_state' :42}
        logger.debug("Starting model building pipeline with params: %s", params)
        train_data = load_data('./data/processed/train_tfidf.csv')
        logger.debug("Preparing training features and labels")
        logger.debug("Columns in train_data: %s", list(train_data.columns))
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)
        logger.debug("Model saved")

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()


