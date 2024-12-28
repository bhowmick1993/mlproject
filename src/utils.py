import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
import dill

def save_object(file_path, obj):
    """
    Save a Python object to a file.
    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The Python object to be saved.
    Returns:
        None
    Raises:
        CustomException: If an error occurs during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(obj, file)
        logging.info("Object saved successfully")
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluates multiple machine learning models on training and test datasets.
    Parameters:
    X_train (array-like or DataFrame): Training feature data.
    y_train (array-like or Series): Training target data.
    X_test (array-like or DataFrame): Test feature data.
    y_test (array-like or Series): Test target data.
    models (dict): A dictionary where keys are model names (str) and values are model instances.
    Returns:
    dict: A dictionary where keys are model names and values are the R^2 score on the test data.
    Raises:
    CustomException: If an error occurs during model evaluation.
    """

    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)