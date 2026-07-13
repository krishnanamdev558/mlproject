import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer       ## to setup the pipeleine
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:     #return path
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_Transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):      #convert categorical into numerical, perform standardscaler and etc
        # This funtion is responsible for data transformation 
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", "race_ethnicity", "lunch", "test_preparation_course"
            ]

            # pipeline to run on training dataset for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),      # handle missing values
                    ("scaler", StandardScaler())    # handling standard scling
                ]
            )
            # pipeline to run on training dataset for categorical features
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),    # replace missing values with mode
                    ("one_hot_encoder", OneHotEncoder()),       # do encoding
                    ("scaler", StandardScaler(with_mean=False))        # do scaling

                ]
            )
            logging.info(f"Numerical columns standard scaling completed: {numerical_columns}")
            logging.info(f"categorical columns encoding completed: {categorical_columns}")

            # combining num and cat ppelines together
            Preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            return Preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")

            logging.info("obtaining preprocessing object")      ## changes

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_features_train_df = train_df.drop(columns=[target_column_name])
            target_features_train_df = train_df[target_column_name]
            # input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            # target_features_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name])
            target_features_test_df = test_df[target_column_name]
            # input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            # target_features_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_features_train_df)         ## concatinatinf input and targetfeatures
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_features_test_df)         ## concatinatinf input and targetfeatures
            ]

            logging.info(f"Saved preprocessing object")

            save_object(        
                ## function used for saving pickle file
                file_path=self.data_Transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_Transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            


            

         

