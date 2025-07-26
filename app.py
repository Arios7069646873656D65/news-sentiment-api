from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("model.pkl")
app = FastAPI()

class  Headline(BaseModel):
    headline: str

@app.post("/predict")
def predict(data: Headline):
    prediction = model.predict([data.headline])[0]

    if prediction == "positive":
        output = "This looks like good news"
    elif prediction == "negative":
        output = "This looks like bad news"
    else:
        output = "This looks neutral"

    return {
        "sentiment": prediction,
        "message": output
    }