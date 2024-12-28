import os
from  dataclasses import dataclass
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join('artifacts', 'preprocesser.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["reading_score","writing_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
                ]
            """
            The below pipeline is doing 2 imortant things:
            1. It is filling the missing values with the median value
            2. It is standardizing the numerical values
            """
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())
                ]
            )

            """
            The below pipeline is doing 3 important things:
            1. It is filling the missing values with the most frequent value
            2. It is one hot encoding the categorical values
            3. It is standardizing the categorical values
            """
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder())
                ]
            )

            logging.info("Numerical pipeline is completed")
            logging.info("Categorical columns encoding completed")

            # combining the two pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test Data read successfully")

            logging.info("Obtaining the preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"
            numerical_columns = ["reading_score","writing_score"]

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocesser_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


