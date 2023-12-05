import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
lasso_model=pickle.load(open('models/insurance_lasso.pkl','rb'))
standard_scaler=pickle.load(open('models/insurance_scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        age = int(request.form.get('age'))
        sex = int(request.form.get('sex'))
        bmi = float(request.form.get('bmi'))
        children = int(request.form.get('children'))
        smoker = int(request.form.get('smoker'))
        region = int(request.form.get('region'))

        new_data_scaled=standard_scaler.transform([[age,sex,bmi,children,smoker,region]])
        result=lasso_model.predict(new_data_scaled)

        return render_template('home.html',age=age,sex=sex,bmi=bmi,children=children,smoker=smoker,region=region,result=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
