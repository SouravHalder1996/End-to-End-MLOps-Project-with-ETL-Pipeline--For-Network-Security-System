from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from src.utils.helper.utils import read_yaml_file, write_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from scipy.stats import ks_2samp
import pandas as pd
import os, sys


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            required_num_columns = len(self.schema_config.get("columns", []))
            actual_num_columns = len(dataframe.columns)

            logging.info(f"Required number of columns: {required_num_columns}")
            logging.info(f"Actual number of columns: {actual_num_columns}")

            if actual_num_columns == required_num_columns:
                logging.info(f"Required number of columns are present")
                return True
            logging.error(f"Required number of columns are not present")
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            required_numerical_columns = set(self.schema_config.get("numerical_columns", []))
            actual_numerical_columns = set(dataframe.columns)
            
            missing_columns = required_numerical_columns - actual_numerical_columns
            
            logging.info(f"Required numerical columns: {required_numerical_columns}")
            logging.info(f"Actual numerical columns in dataframe: {actual_numerical_columns}")
            
            if missing_columns:
                logging.error(f"Missing numerical columns: {missing_columns}")
                return False
            logging.info("All required numerical columns are present")
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold = 0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column:{
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            logging.info("Validating number of columns in training data")
            status = self.validate_number_of_columns(train_df)
            if not status:
                error_message = "Training data does not have the required number of columns \n"
            
            logging.info("Validating number of columns in testing data")
            status = self.validate_number_of_columns(test_df)
            if not status:
                error_message = "Testing data does not have the required number of columns \n"
            
            logging.info("Validating numerical columns in training data")
            status = self.validate_numerical_columns(train_df)
            if not status:
                error_message = "Training data is missing required numerical columns \n"
            
            logging.info("Validating numerical columns in testing data")
            status = self.validate_numerical_columns(test_df)
            if not status:
                error_message = "Testing data is missing required numerical columns \n"
            
            status = self.detect_data_drift(base_df=train_df, current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_df.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)