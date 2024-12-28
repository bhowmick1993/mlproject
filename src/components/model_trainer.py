import os
import sys
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_mode_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.molel_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        Initiates the model training process by training multiple models and selecting the best one based on R2 score.
        Parameters:
        train_array (numpy.ndarray): The training data array where the last column is the target variable.
        test_array (numpy.ndarray): The testing data array where the last column is the target variable.
        preprocessor_path (str): The path to the preprocessor object.
        Returns:
        float: The R2 score of the best model on the test data.
        Raises:
        CustomException: If no model achieves an R2 score of at least 0.6 or if any other exception occurs during the process.
        """
        try:
            logging.info("Entering the model training component")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # now we are creating a dictionary of models
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor()   
            }

            """
            evaluate_model is created in the utils.py file
            """
            models_report : dict = evaluate_model(X_train = X_train, y_train = y_train, 
                                                  X_test = X_test, y_test = y_test, 
                                                  models = models)
            
            # get the best model score from the dict
            best_model_score = max(sorted(models_report.values()))

            # get the best model name
            best_model_name = [model_name for model_name, model_score in models_report.items() if model_score == best_model_score][0]

            # get the best model
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            
            logging.info("Best model found is {0} with R2 score {1}".format(best_model_name, best_model_score))

            # save the best model
            save_object(self.molel_trainer_config.trained_mode_file_path, 
                        best_model)
            
            predicted = best_model.predict(X_test)
            r2_score_for_model = r2_score(y_test, predicted)

            return r2_score_for_model

        except Exception as e:
            raise CustomException(e, sys)
