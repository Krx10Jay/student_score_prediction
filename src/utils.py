import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj): # creating a function that creates a folder and saves it in a desired location
    try:
        dir_path = os.path.dirname(file_path)# extracting the name of the directory where you file is saved

        os.makedirs(dir_path, exist_ok=True)# creating a directory

        with open(file_path, "wb") as file_obj: # opens the location
            pickle.dump(obj, file_obj) # appends your file

    except Exception as e:
        raise CustomException(e, sys)
    


    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models,param=None):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f'performing gridsearch for {model_name}...')
            grid_search= GridSearchCV(model,param[model_name], cv=5)

            grid_search.fit(X_train,y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_test, y_test_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score


        return report, test_model_score

    except Exception as e:
        raise CustomException(e, sys)