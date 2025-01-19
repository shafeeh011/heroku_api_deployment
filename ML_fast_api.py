from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
from regex import P
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI()

class model_input(BaseModel):
    
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int
    
# for predicting the result
diabetes_model = pickle.load(open('/home/muhammed-shafeeh/AI_ML/Ai-and-Ml/ML_diabetes/diabetes_model.sav', 'rb'))

@app.post('/diabetes_prediction')

def diabetes_prediction(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dectionary = json.loads(input_data)
    
    # conver the dict to list 
    preg = input_dectionary['Pregnancies']
    glu = input_dectionary['Glucose']
    bp = input_dectionary['BloodPressure']
    skin = input_dectionary['SkinThickness']
    insulin = input_dectionary['Insulin']
    bmi = input_dectionary['BMI']
    dpj = input_dectionary['DiabetesPedigreeFunction']
    age = input_dectionary['Age']
    
    input_list = [preg, glu, bp, skin, insulin, bmi, dpj, age]
    
    # loading the saved model

        
    #input_list = [1,85,66,29,0,26.6,0.351,31]
    #reshape the data as we are predicting for one instance
    input_list_as_numpy_array = np.asarray(input_list)
    input_data_reshaped = input_list_as_numpy_array.reshape(1,-1)
    #print(input_data_reshaped.shape)
    #print(input_data_reshaped)
    
    # standrdizing the input data
    # for standrdizing the input data
    X = pickle.load(open('/home/muhammed-shafeeh/AI_ML/Ai-and-Ml/ML_diabetes/diabetes_X.sav', 'rb'))
    #print(X)
    scaler = StandardScaler()
    scaler.fit(X)
    std_data = scaler.transform(input_data_reshaped)
    #print(std_data.shape)
    #print(std_data)
    
    
    # prediction
    try:
        prediction = diabetes_model.predict(std_data)
        #print(prediction)
    except Exception as e:
        return {'error': str(e)}
    
    if prediction[0]==0:
        #print(prediction)
        return 'The Person is not Diabetic'
        
    
    else:
        return 'The Person is Diabetic'
    
