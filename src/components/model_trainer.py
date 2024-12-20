import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import (
    save_object,
    evaluate_models
)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('<<<<<<  MODEL-TRAINER  >>>>>>')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'Linear Regression':LinearRegression(),
                'K-Neighbours Regressor':KNeighborsRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Random Forest':RandomForestRegressor(),
                'AdaBoost Regressor':AdaBoostRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'CatBoost Regressor':CatBoostRegressor(verbose=False),
                'XGBoost Regressor':XGBRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    'max_depth':[1,2,3,4,5,6,7,8,10,11,12]
                    # 'max_features':['sqrt','log2'],
                },
                'K-Neighbours Regressor': {
                    'n_neighbors': [5,7,9,11,13,15],
                    'weights' : ['uniform','distance'],
                    'metric' : ['minkowski','euclidean','manhattan']
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            models_report=evaluate_models(X_train,y_train,X_test,y_test,models,params)
            logging.info('TRAINING: Model Completed Training')

            ## get best model score from model_report dictionary
            best_model_score=max(sorted(models_report.values()))

            ## get best model name from dictionary
            best_model_name=list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            if best_model_score<0.6:
                raise CustomException('No best model found')

            best_model=models[best_model_name]

            logging.info('Best Model Found On Both Training & Testing Dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Saved Model object')

            logging.info(models_report)

            predicted=best_model.predict(X_test)
            r_square=r2_score(y_test,predicted)

            return best_model_name,r_square

        except Exception as e:
            raise CustomException(e,sys)
