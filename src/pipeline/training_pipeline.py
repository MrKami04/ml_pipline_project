import os , sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass
from src.utils import save_object
from src.utils import evaluate_model




if __name__ == "__main__":
    
    try:
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initialize_data_transformation(
            train_path, test_path)    
        
        
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    except Exception as e:
        logging.info('Error in training pipeline')
        raise CustomException(e, sys)