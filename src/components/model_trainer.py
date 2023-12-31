import os
import sys

from ball.Lib.dataclasses import dataclass
from src.logger import logging
from src.utils import save_object,evaluate_models
from src.exceptions import CustomException

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor
)

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

@dataclass
class modeltrainerconfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class modeltrainer:
    def __init__(self):
        self.model_trainer_config = modeltrainerconfig()

    def initiate_model_training(self,train_arr,test_arr):
        
        try:
            logging.info("splitting data ")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
        
            models = {
                "DescisionTree" : DecisionTreeRegressor(),
                "RandomForest"  : RandomForestRegressor(),
                "Catboost"      : CatBoostRegressor(verbose=False),
                "XGboost"       : XGBRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighbours"   : KNeighborsRegressor(),
                "GradientBossting" : GradientBoostingRegressor(),
                "AdaBoost"       : AdaBoostRegressor() 
            }

            # params = {
            #     "DescisionTree" : {
            #             'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            #             'splitter':['best','random'],
            #             'max_features':['sqrt','log2']
            #     },
            #     "RandomForest":{
            #             'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        
            #             'max_features':['sqrt','log2',None],
            #             'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "CatBoosting Regressor":{
            #             'depth': [6,8,10],
            #             'learning_rate': [0.01, 0.05, 0.1],
            #             'iterations': [30, 50, 100]
            #     },
            #     "XGBRegressor":{
            #             'learning_rate':[.1,.01,.05,.001],
            #             'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "Linear Regression":{},
            #     "Gradient Boosting":{
            #             'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
            #             'learning_rate':[.1,.01,.05,.001],
            #             'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
            #             'criterion':['squared_error', 'friedman_mse'],
            #             'max_features':['auto','sqrt','log2'],
            #             'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "AdaBoost Regressor":{
            #         'learning_rate':[.1,.01,0.5,.001],
            #         'loss':['linear','square','exponential'],
            #         'n_estimators': [8,16,32,64,128]
            #     }   
            # }

            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models) #add params if u want more accuracy
        
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("best model found")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            predicted = best_model.predict(x_test)

            r2_squared = r2_score(y_test,predicted)

            return r2_squared,best_model_name,model_report

        except Exception as e:
            raise CustomException(e,sys)















