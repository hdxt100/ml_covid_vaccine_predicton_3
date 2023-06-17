import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging


from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
 
            numerical_columns = ['covid_worry', 'covid_awareness', 'antiviral_medication',
            'contact_avoidance', 'bought_face_mask', 'wash_hands_frequently',
            'avoid_large_gatherings', 'reduced_outside_home_cont',
            'avoid_touch_face', 'chronic_medic_condition',
            'cont_child_undr_6_mnths', 'is_health_worker',
            'is_covid_vacc_effective', 'is_covid_risky', 'sick_from_covid_vacc',
            'is_seas_vacc_effective', 'is_seas_risky', 'sick_from_seas_vacc',
            'no_of_adults', 'no_of_children']
            categorical_columns = ['age_bracket', 'qualification', 'address', 'sex', 'marital_status',
            'employment', 'status']
            
            num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler(with_mean=False))
            
            ])
            cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ("one_hot_encoder",OneHotEncoder()),
            ('std_scaler', StandardScaler(with_mean=False))
                  
            ])
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline", cat_pipeline,categorical_columns)
                
                ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            train_df.drop(['unique_id','has_health_insur', 'income_level', 'dr_recc_covid_vacc','dr_recc_seasonal_vacc', 'housing_status'], axis=1,inplace=True)
            test_df.drop(['unique_id','has_health_insur', 'income_level', 'dr_recc_covid_vacc','dr_recc_seasonal_vacc', 'housing_status'], axis=1,inplace=True)
            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)
            
            train_df['no_of_children'] = np.where(train_df['no_of_children']>2, 2, train_df['no_of_children'])
            train_df['no_of_adults'] = np.where(train_df['no_of_adults'] > 2, 2, train_df['no_of_adults'])
            train_df['is_seas_vacc_effective'] = np.where(train_df['is_seas_vacc_effective']>3,3, train_df['is_seas_vacc_effective'])

            test_df['no_of_children'] = np.where(test_df['no_of_children']>2, 2, test_df['no_of_children'])
            test_df['no_of_adults'] = np.where(test_df['no_of_adults'] > 2, 2, test_df['no_of_adults'])
            test_df['is_seas_vacc_effective'] = np.where(test_df['is_seas_vacc_effective']>3,3, test_df['is_seas_vacc_effective'])
            
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            
        
            
            target_column_name='covid_vaccine'
            

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
