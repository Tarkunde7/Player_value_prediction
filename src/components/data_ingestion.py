import os
import sys
from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from ball.Lib.dataclasses import dataclass

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
            logging.info("reading and concating the data")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)

            logging.info("Train Test Split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=2)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)

            logging.info("Ingestion of Data is complete")

            return(
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()