from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        logging.info('Saving object initiated')
        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info('Object saved successfully')
        
    except Exception as e:
        logging.info('Error in saving object')
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        logging.info('Loading object initiated')
        
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
            
        logging.info('Object loaded successfully')
        return obj
    
    except Exception as e:
        logging.info('Error in loading object')
        raise CustomException(e, sys)
    
    

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        logging.info('Model evaluation initiated')
        report = {}

        for model_name, model in models.items():
            para = params[model_name]

            gs = GridSearchCV(model, para, cv=5)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_acc

        logging.info('Model evaluation completed')
        return report

    except Exception as e:
        logging.info('Error in model evaluation')
        raise CustomException(e, sys)
    
    
    
    
    
    
    
    
    
    
    
    