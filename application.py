from flask import Flask,request,jsonify,render_template
import numpy as np
from  sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle


application=Flask(__name__)
app=application 


#import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
Standard_Scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("home.html")



@app.route('/prediction',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        DC=float(request.form.get('DC'))
        ISI=float(request.form.get('ISI'))
        BUI=float(request.form.get('BUI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))    

        new_data=Standard_Scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes,Region]])
        result=ridge_model.predict(new_data)

        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')
    


if __name__=="__main__":
    app.run(host="0.0.0.0")