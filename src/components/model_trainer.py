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

            models_report=evaluate_models(X_train,y_train,X_test,y_test,models)
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

            predicted=best_model.predict(X_test)
            r_square=r2_score(y_test,predicted)

            return best_model_name,r_square

        except Exception as e:
            raise CustomException(e,sys)
