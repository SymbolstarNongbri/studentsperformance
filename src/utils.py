import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import dill

from src.exception import CustomException
from src.logger import logging




def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    '''
    ::::  This is the function to evaluate models  ::::
    '''
    try:
        report={}

        # How many number of linear-models?
        models_len=len(list(models))

        logging.info('TRAINING: Evaluating Best Model')
        for i in range(models_len):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]

            logging.info(f"-----> CV - {model} ")
            gs = GridSearchCV(model,para,cv=3)
            g_res=gs.fit(X_train,y_train)
            logging.info(g_res.best_params_)

            model.set_params(**g_res.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        logging.info('TRAINING: Completed Evaluating Best Model')
        return report

    except Exception as e:
        raise CustomException(e,sys)