## contains common functions used in project

import os
import sys

import numpy as np
import dill
import pandas as pd

from src.exception import CustomException

from sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains each model, evaluates it using R² score,
    and returns a dictionary of model scores.
    """
    try:
        report = {}
        for i in range(len(list(models))):
            
            model = list(models.values())[i]

            # Train the model
            model.fit(X_train, y_train)

            # Predict on train data
            y_train_pred=model.predict(X_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate R² score for train
            train_model_score = r2_score(y_train, y_train_pred)

            # Calculate R² score for test
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the score
            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path, obj):
    """
    Save a Python object to disk using pickle.

    Args:
        file_path (str): Path where the object will be saved.
        obj: Any Python object to be serialized.
    """
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)