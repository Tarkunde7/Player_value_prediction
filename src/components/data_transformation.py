import sys
import os
from ball.Lib.dataclasses import dataclass

import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            numerical_columns = ['age','height_cm','weight_kg','overall','potential','value_eur','international_reputation','weak_foot','pace','shooting','passing','dribbling','physic','defending','gk_diving','gk_handling','gk_positioning','gk_kicking','gk_reflexes','gk_speed']
            categorical_columns = ['body_type']

            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy= "constant")),
                ("scaler",StandardScaler())    
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ]
            )

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ],
                remainder= 'passthrough'
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading test and train data complete")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.copy()  # Keep all columns
            target_feature_train_df = train_df['value_eur']

            input_feature_test_df = test_df.copy()  # Keep all columns
            target_feature_test_df = test_df['value_eur']

            logging.info("Applying preprocessing data on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Remove 'value_eur' column from input features
            input_feature_train_arr = input_feature_train_arr[:, :-1]
            input_feature_test_arr = input_feature_test_arr[:, :-1]

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)