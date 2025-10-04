import os, sys
import json
from dotenv import load_dotenv
import certifi
import pymongo
import pandas as pd
# from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca = certifi.where()

class NetworkDataETL():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def convert_csv_to_json(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)
            records = list(json.loads(df.T.to_json()).values())
            return records
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def load_into_mongodb(self, records, db_name, collection_name):
        try:
            self.records = records
            self.db_name = db_name
            self.collection_name = collection_name

            self.pymongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.database = self.pymongo_client[self.db_name]
            self.collection = self.database[self.collection_name]

            self.collection.insert_many(self.records)
            print(f"Data inserted successfully into MONGODB database, record count: {len(self.records)}")
                
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

if __name__ == "__main__":
    FILE_PATH = "../data/phisingData.csv"
    DB_NAME = os.getenv("MONGO_DB_NAME")
    COLLECTION_NAME = os.getenv("MONGO_DB_COLLECTION_NAME")

    etl = NetworkDataETL()
    records = etl.convert_csv_to_json(file_path=FILE_PATH)
    etl.load_into_mongodb(records=records, db_name=DB_NAME, collection_name=COLLECTION_NAME)
