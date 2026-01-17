import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("spam_model.pkl")

app = FastAPI(title="Spam Classifier API")


class Message(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Spam Classifier API is running"}


@app.post("/predict")
def predict(message: Message):
    prob = model.predict_proba([message.text])[0][1]
    prediction = 1 if prob > 0.4 else 0
    label = "Spam" if prediction == 1 else "Not Spam"
    return {"prediction": label}
