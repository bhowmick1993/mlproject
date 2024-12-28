import sys
import pandas as pd
import numpy as np
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocesser_path = 'artifacts/preprocesser.pkl'
            model = load_object(file_path = model_path)
            preprocesser = load_object(file_path = preprocesser_path)
            data_scaled = preprocesser.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Used for mapping all the data that we will be getting from the user through HTML form
    """
    def __init__(self,
                 gender : str,
                 race_ethinicity : str,
                 parental_level_of_education : str,
                 lunch : str,
                 test_preparation_course : str,
                 reading_score : int,
                 writing_score : int):
        
        self.gender = gender
        self.race_ethinicity = race_ethinicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        This method will convert the data into a pandas dataframe
        """
        try:
            custom_data_input_dict = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethinicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)

        