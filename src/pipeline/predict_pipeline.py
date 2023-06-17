import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))



import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        covid_worry : int,
        covid_awareness : int,
        antiviral_medication: int,
        contact_avoidance: int,
        bought_face_mask: int,
        wash_hands_frequently: int,
        avoid_large_gatherings: int,
        reduced_outside_home_cont: int,
        avoid_touch_face: int,
        chronic_medic_condition: int,
        cont_child_undr_6_mnths: int,
        is_health_worker: int,
        is_covid_vacc_effective: int,
        is_covid_risky: int,
        sick_from_covid_vacc: int,
        is_seas_vacc_effective: int,
        is_seas_risky: int,
        sick_from_seas_vacc: int,
        no_of_adults: int,
        no_of_children: int,
        age_bracket :str,
        qualification :str,
        address :str,
        sex :str,
        marital_status :str,
        employment :str,
        status :str):

        self.covid_worry = covid_worry

        self.covid_awareness = covid_awareness

        self.antiviral_medication = antiviral_medication

        self.contact_avoidance = contact_avoidance

        self.bought_face_mask = bought_face_mask

        self.wash_hands_frequently = wash_hands_frequently

        self.avoid_large_gatherings = avoid_large_gatherings
        
        self.reduced_outside_home_cont = reduced_outside_home_cont
        
        self.avoid_touch_face = avoid_touch_face
        
        self.chronic_medic_condition = chronic_medic_condition
        
        self.cont_child_undr_6_mnths = cont_child_undr_6_mnths
        
        self.is_health_worker = is_health_worker
        
        self.is_covid_vacc_effective = is_covid_vacc_effective
        self.is_covid_risky = is_covid_risky
        self.sick_from_covid_vacc = sick_from_covid_vacc
        self.is_seas_vacc_effective = is_seas_vacc_effective
        self.is_seas_risky = is_seas_risky
        self.sick_from_seas_vacc = sick_from_seas_vacc
        self.no_of_adults = no_of_adults
        self.no_of_children = no_of_children
        self.age_bracket = age_bracket
        self.qualification = qualification
        self.address = address
        self.sex = sex
        self.marital_status = marital_status
        self.employment = employment
        self.status = status
        
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "covid_worry": [self.covid_worry],
                "covid_awareness": [self.covid_awareness],
                "antiviral_medication": [self.antiviral_medication],
                "contact_avoidance": [self.contact_avoidance],
                "bought_face_mask": [self.bought_face_mask],
                "wash_hands_frequently": [self.wash_hands_frequently],
                "avoid_large_gatherings": [self.avoid_large_gatherings],
                "reduced_outside_home_cont": [self.reduced_outside_home_cont],
                "avoid_touch_face": [self.avoid_touch_face],
                "chronic_medic_condition": [self.chronic_medic_condition],
                "cont_child_undr_6_mnths": [self.cont_child_undr_6_mnths],
                "is_health_worker": [self.is_health_worker],
                "is_covid_vacc_effective": [self.is_covid_vacc_effective],
                "is_covid_risky": [self.is_covid_risky],
                "sick_from_covid_vacc": [self.sick_from_covid_vacc],
                "is_seas_vacc_effective": [self.is_seas_vacc_effective],
                "is_seas_risky": [self.is_seas_risky],
                "sick_from_seas_vacc": [self.sick_from_seas_vacc],
                "no_of_adults": [self.no_of_adults],
                "no_of_children": [self.no_of_children],
                "age_bracket": [self.age_bracket],
                "qualification": [self.qualification],
                "address": [self.address],
                "sex": [self.sex],
                "marital_status": [self.marital_status],
                "employment": [self.employment],
                "status": [self.status]
            
           }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
