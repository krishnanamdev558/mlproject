## contains common functions used in project

import os
import sys

import numpy as np
import dill
import pandas as pd

from src.exception import CustomException

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