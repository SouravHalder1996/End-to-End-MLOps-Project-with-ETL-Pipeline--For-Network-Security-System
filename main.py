from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.entity.config_entity import TrainingPipelineConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
import sys

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Starting data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed successfully {data_ingestion_artifact}")

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Starting data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data validation completed successfully {data_validation_artifact}")

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Starting data transformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data transformation completed successfully {data_transformation_artifact}")

    except Exception as e:
        raise NetworkSecurityException(e, sys)