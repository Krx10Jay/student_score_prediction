import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
#from src.components import data_injection
from src.utils import save_object


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationPath:
    preprocessor_obj_path:str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self, train_data_path:str,test_data_path:str,target_col:str):
        self.data_transformer_path= DataTransformationPath()
        self.train_data_path =train_data_path
        self.test_data_path =test_data_path
        self.target_col_name =target_col

    def get_data_transformer(self):#train_data_path,test_data_path,target_col): #data_path):
        # the structure for transformation

        try:
            #self.train_data_path =train_data_path
            #self.test_data_path =test_data_path
            #self.target_col_name =target_col

            logging.info('loading data...')
            df=pd.read_csv(self.train_data_path)
            # separating the numeric features from the categorical ones
            numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O' and feature != 'math_score']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O' and feature != 'math_score']
            logging.info('feature splitting completed')
            logging.info(f"numerical cols: {numeric_features}")
            logging.info(f"categorical cols: {categorical_features}")

            # creating a pipeline for replacing Na values and scaling the data
            num_pipeline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())
                    ]
            )

            cat_pipline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                     ]
            )

            # merging the results of the two pipelines into a dataframe
            preprocessor = ColumnTransformer(
                [('numerical', num_pipeline,numeric_features),
                ('Categorical', cat_pipline,categorical_features)]
            )

            logging.info('Preprocessor Structure created')

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformer(self): #train_path, test_path, target_col):
        logging.info('getting the train and test data')
        
        try:
            self.train_data_path
            self.test_data_path 
            self.target_col_name
            train_df= pd.read_csv(self.train_data_path)
            test_df= pd.read_csv(self.test_data_path)

            preprocessor_obj = self.get_data_transformer()#(self.train_data_path,self.test_data_path,'math_score')

            target_col_name= self.target_col_name

            train_X = train_df.drop(target_col_name, axis=1)
            train_y = train_df[target_col_name]

            test_X =test_df.drop(target_col_name,axis=1)
            test_y = test_df[target_col_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            train_X_arr=preprocessor_obj.fit_transform(train_X)
            test_X_arr=preprocessor_obj.transform(test_X)


            train_arr = np.c_[
                train_X_arr, np.array(train_y)
            ]
            test_arr = np.c_[test_X_arr, np.array(test_y)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformer_path.preprocessor_obj_path,
                obj=preprocessor_obj
                )
            
            return (
                train_arr,
                test_arr,
                self.data_transformer_path.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomException(e,sys)