import yaml
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
import os, sys
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info(f"Saving object to file: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully to file: {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        fitted_models = {}

        for model_name in models.keys():
            print(f"\n{'='*50}")
            print(f"Training {model_name}...")
            print(f"{'='*50}")

            model = models[model_name]
            param = params[model_name]

            gs = GridSearchCV(
                estimator=model,
                param_grid=param,
                scoring='f1',
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            fitted_models[model_name] = best_model

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = f1_score(y_train, y_train_pred)
            test_model_score = f1_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            print(f"{model_name} - Best Params: {gs.best_params_}")
            print(f"{model_name} - Train F1 Score: {train_model_score:.4f}")
            print(f"{model_name} - Test F1 Score: {test_model_score:.4f}")
            logging.info(f"{model_name} - Best Params: {gs.best_params_}")
            logging.info(f"{model_name} - Train F1 Score: {train_model_score:.4f}")
            logging.info(f"{model_name} - Test F1 Score: {test_model_score:.4f}")
            
        return report, fitted_models
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)