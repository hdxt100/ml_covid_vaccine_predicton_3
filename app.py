from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
	           covid_worry=request.form.get('covid_worry'),	
	            covid_awareness=request.form.get('covid_awareness'),	
            antiviral_medication=request.form.get('antiviral_medication'),	
            contact_avoidance=request.form.get('contact_avoidance'),	
            bought_face_mask=request.form.get('bought_face_mask'),	
            wash_hands_frequently=request.form.get('wash_hands_frequently'),	
            avoid_large_gatherings=request.form.get('avoid_large_gatherings'),	
            reduced_outside_home_cont=request.form.get('reduced_outside_home_cont'),	
            avoid_touch_face=request.form.get('avoid_touch_face'),	
            chronic_medic_condition=request.form.get('chronic_medic_condition'),	
            cont_child_undr_6_mnths=request.form.get('cont_child_undr_6_mnths'),	
            is_health_worker=request.form.get('is_health_worker'),	
            is_covid_vacc_effective=request.form.get('is_covid_vacc_effective'),	
            is_covid_risky=request.form.get('is_covid_risky'),	
            sick_from_covid_vacc=request.form.get('sick_from_covid_vacc'),	
            is_seas_vacc_effective=request.form.get('is_seas_vacc_effective'),	
            is_seas_risky=request.form.get('is_seas_risky'),	
            sick_from_seas_vacc=request.form.get('sick_from_seas_vacc'),	
            no_of_adults=request.form.get('no_of_adults'),	
            no_of_children=request.form.get('no_of_children'),	
            age_bracket=request.form.get('age_bracket'),	
            qualification=request.form.get('qualification'),	
            address=request.form.get('address'),	
            sex=request.form.get('sex'),	
            marital_status=request.form.get('marital_status'),	
            employment=request.form.get('employment'),	
            status=request.form.get('status')
            
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=9999)        
