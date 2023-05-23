# import onnx as onnx
from fastapi import FastAPI
import uvicorn
import os
import pickle
import joblib
from pydantic import BaseModel
from preprocessing import TextPreprocessor
# Declaring our FastAPI instance

app = FastAPI()

text_preprocessor = TextPreprocessor()
counrty_dict = {"EG": "مصري", "LY": "ليبي", "LB": "لباني" , "SD":"سوداني" ,"MA":"مغربي"}

class request_body(BaseModel):
    tweet1 : str
CWD = os.getcwd()
PWD = os.path.dirname(CWD)
models_folder_path = PWD + "/models/"
SVM_PATH = models_folder_path + 'svm_model.joblib'
#RNN_PATH = models_folder_path + 'bilstm2_model.joblib'
model = joblib.load(SVM_PATH)
# text=""
@app.post('/predict')
def predict(data : request_body):
    test_data = [
            data.tweet1
    ]
    trans_data = text_preprocessor.transform(test_data)
    if trans_data[0] == "" or trans_data[0] == " ":
        text = "من فضلك أدخل جملة عربية صحيحة"
    else:
        predict = model.predict(trans_data)
        for i in counrty_dict:
            if i == predict:
                text = counrty_dict[i]
    return { 'الاجابة هي :{}'.format(text)}






