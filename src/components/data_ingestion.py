import os
import sys
from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from ball.Lib.dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import modeltrainer
from src.components.model_trainer import modeltrainerconfig

os.environ["LOKY_MAX_CPU_COUNT"] = "4" #for increasing the number of cpu cores

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion has started")
        try:
            file_paths = [
                r'E:\FootballProject\Data\players_16renew.csv',
                r'E:\FootballProject\Data\players_17renew.csv',
                r'E:\FootballProject\Data\players_18renew.csv',
                r'E:\FootballProject\Data\players_19renew.csv',
                r'E:\FootballProject\Data\players_20renew.csv'
            ]

            dataframes = []
            for file_path in file_paths:
                data = pd.read_csv(file_path)
                dataframes.append(data)
            
            df = pd.concat([dataframes[0],dataframes[1],dataframes[2],dataframes[3],dataframes[4]],ignore_index= True)
            df = df[['age','height_cm','weight_kg','overall','potential','value_eur','international_reputation','weak_foot','body_type','pace','shooting','passing','dribbling','physic','defending','gk_diving','gk_handling','gk_positioning','gk_kicking','gk_reflexes','gk_speed']].copy()
            
            logging.info("reading ,concating and selecting specific features from the data completed")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)

            logging.info("Train Test Split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=2)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)

            logging.info("Ingestion of Data is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = modeltrainer()
    accuracy,model_name,models_report = modeltrainer.initiate_model_training(train_arr,test_arr)
    for key,value in models_report.items():
        print(f"{key} -> accuracy% -> {(value)*100}%")
    print(f"From all the 8 models used on a data of shape(86542,21) the best model is {model_name} and its accuracy is : {(accuracy)*100}%")