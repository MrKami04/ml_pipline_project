# create predtion pipeline class
# create function for load a object
# create custom class based upon our dataset
# create function to convert data into dataframe with the help of dictionary

import os , sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts/model_trainer','model.pkl')
            preprocessor_path = os.path.join('artifacts/data_transformation','preprocessor.pkl')
            
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info('Error in prediction pipeline')
            raise CustomException(e, sys)
        
        
class CustomClass:
    def __init__(self, age:int, workclass:int, education_num:int, marital_status:int,
                 occupation:int, relationship:int, race:int, sex:int, capital_gain:int, 
                 capital_loss:int, hours_per_week:int, native_country:int):
        
        self.age = age
        self.workclass = workclass
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain    
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week    
        self.native_country = native_country
        
    def custom_data_frame(self):
        try:
            custom_data_input_dict = {
                'age': [self.age],
                'workclass': [self.workclass],
                'education_num': [self.education_num],
                'marital_status': [self.marital_status],
                'occupation': [self.occupation],
                'relationship': [self.relationship],
                'race': [self.race],
                'sex': [self.sex],
                'capital_gain': [self.capital_gain],
                'capital_loss': [self.capital_loss],
                'hours_per_week': [self.hours_per_week],
                'native_country': [self.native_country]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.info('Error in creating custom data frame')
            raise CustomException(e, sys)