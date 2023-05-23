from fastapi import FastAPI, Request
import uvicorn
import os
import pickle
import joblib
from pydantic import BaseModel
from preprocessing import TextPreprocessor
from fastapi.templating import Jinja2Templates

# Declaring our FastAPI instance
app = FastAPI()
templates = Jinja2Templates(directory="templates")

text_preprocessor = TextPreprocessor()
country_dict = {"EG": "مصري", "LY": "ليبي", "LB": "لباني", "SD": "سوداني", "MA": "مغربي"}

class PredictionResult(BaseModel):
    prediction: str

class InputText(BaseModel):
    text: str

CWD = os.getcwd()
PWD = os.path.dirname(CWD)
models_folder_path = PWD + "/models/"
SVM_PATH = models_folder_path + 'svm_model.joblib'
model = joblib.load(SVM_PATH)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input_text: InputText):
    text = input_text.text.strip()
    if not text:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Please enter validSure, here's an example of how you can modify the FastAPI server code to take input from the user through a user interface:

```python
from fastapi import FastAPI, Request
import uvicorn
import os
import pickle
import joblib
from pydantic import BaseModel
from preprocessing import TextPreprocessor
from fastapi.templating import Jinja2Templates

# Declaring our FastAPI instance
app = FastAPI()
templates = Jinja2Templates(directory="templates")

text_preprocessor = TextPreprocessor()
country_dict = {"EG": "مصري", "LY": "ليبي", "LB": "لباني", "SD": "سوداني", "MA": "مغربي"}

class PredictionResult(BaseModel):
    prediction: str

class InputText(BaseModel):
    text: str

CWD = os.getcwd()
PWD = os.path.dirname(CWD)
models_folder_path = PWD + "/models/"
SVM_PATH = models_folder_path + 'svm_model.joblib'
model = joblib.load(SVM_PATH)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input_text: InputText):
    text = input_text.text.strip()
    if not text:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Please enter validArabic text."})
    else:
        test_data = [text]
        trans_data = text_preprocessor.transform(test_data)
        prediction = model.predict(trans_data)
        country = country_dict.get(prediction[0], "Unknown")
        return templates.TemplateResponse("index.html", {"request": request, "prediction": country})
