# Handle missin values
# outliers treatment
# Handle imbalanced dataset
# convert categorical to numerical


import os , sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts/data_transformation','preprocessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        
    def get_data_transformer_object(self):
        try:
            logging.info('Data transformation initiated')
            # Define which columns should be ordinal and which should be nominal
            # Define the custom ranking for the ordinal variable
            # ordinal variable
            # nominal variable
            numerical_cols = ['age', 'workclass', 'education_num', 'marital_status',
                              'occupation','relationship', 'race', 'sex', 'capital_gain',
                              'capital_loss','hours_per_week',"native_country"]
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols)
            ])
            return preprocessor

        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)

    def remove_outliers_IQR(self, df, column):
        try:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df.loc[(df[column] > upper_bound), column] = upper_bound
            df.loc[(df[column] < lower_bound), column] = lower_bound
            return df
        except Exception as e:
            logging.info(f'Error in removing outliers from column: {column}')
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info('Initializing data transformation')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            numerical_cols = ['age', 'workclass', 'education_num', 'marital_status',
                              'occupation', 'relationship', 'race', 'sex', 'capital_gain',
                              'capital_loss', 'hours_per_week', 'native_country']

            logging.info('Removing outliers from training dataframe')
            for i in numerical_cols:
                train_df = self.remove_outliers_IQR(train_df, i)

            logging.info('Removing outliers from testing dataframe')
            for i in numerical_cols:
                test_df = self.remove_outliers_IQR(test_df, i)

            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_data_transformer_object()
            target_column = 'income'
            drop_columns = [target_column]

            logging.info('Splitting training dataframe into dependent and independent features')
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]

            logging.info('Splitting testing dataframe into dependent and independent features')
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying transformation object on training and testing dataframes')
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_feature_test_df)]

            logging.info('Saving preprocessor object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Error in initializing data transformation')
            raise CustomException(e, sys)