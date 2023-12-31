import pandas as pd 
import numpy as np 
import sys
import os
import dill
from src.exceptions import CustomException
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok= True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models): #add params if u want more accurate model
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            #para = params[list(models.keys())[i]]

            #gs = GridSearchCV(model,cv=3)
            #gs.fit(x_train,y_train)

            #model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e,sys)