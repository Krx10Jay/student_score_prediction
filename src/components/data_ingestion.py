import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformationPath
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer,ModelTrainerConfig



@dataclass
class DataIngestionPath:
    train_data_path:str =os.path.join('artifacts', 'train_data.csv')
    test_data_path:str =os.path.join('artifacts', 'test_data.csv')
    raw_data_path:str =os.path.join('artifacts', 'raw_data.csv')

class DataIngestion: 
    def __init__(self, DataIngestionPath=DataIngestionPath()):
        self.ingestion_path = DataIngestionPath
    
    def initiate_dataingestion(self, data:str):
        # data= the path of the data to be ingested
        logging.info('initiating data Ingestion...')
        try:
            #data = r'Notebook/Data/students_data.csv'
            df=pd.read_csv(data)
            logging.info('read data')

            os.makedirs(os.path.dirname(self.ingestion_path.train_data_path), exist_ok=True)# creating a directory: artifacts
            df.to_csv(self.ingestion_path.raw_data_path, index=False, header=True)

            logging.info('Initiating spliting of data')
            train_df,test_df=train_test_split(df, random_state=45, test_size=.2)
            

            train_df.to_csv(self.ingestion_path.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_path.test_data_path, index=False, header=True)
            logging.info('Data spliting complete')

            return (
                self.ingestion_path.train_data_path,
                self.ingestion_path.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    data = r'Notebook/Data/students_data.csv'
    train,test=DataIngestion().initiate_dataingestion(data)

    DataTransformationPath()
    data_transformation= DataTransformation(train,test,'math_score')
    train_arr,test_arr,_=data_transformation.initiate_data_transformer()

    model_trainer=ModelTrainer()

    model_trainer.initiate_model_trainer(train_arr,test_arr)