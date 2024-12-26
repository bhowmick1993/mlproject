import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass

"""
If we are only defining the class and not using it, 
then we can define the class in the same file where we are defining the dataclass.
"""
@dataclass
class DataIngestionConfig:
    # Inputs to my data ingestion component
    train_data_path: str =  os.path.join('artifacts', 'train.csv')
    test_data_path: str =  os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')    

class DataIngestion:
    def __init__(self):
        self.ingsestion_config = DataIngestionConfig() #this is now a class variable

    def initiate_data_ingestion(self):
        logging.info("Entering the data ingestion component")
        try:
            df = pd.read_csv("C:\AB_Personal\mlproject\data\stud.csv")
            logging.info("Data read successfully")

            os.makedirs(os.path.dirname(self.ingsestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingsestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split is initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingsestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingsestion_config.test_data_path, index=False, header=True)

            logging.info("Train test split is completed")

            return(
                self.ingsestion_config.train_data_path,
                self.ingsestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
        


