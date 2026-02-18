from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict
import joblib

app= FastAPI()
model=joblib.load('P:/project/pythonpro/myvenv/fl-analysis/models/model.pkl')

def make_prediction(data: pd.DataFrame) -> List[int]:
    # Make predictions using the loaded model
    predictions = model.predict(data)
    return predictions.tolist()
class DataInput(BaseModel):
    data: List[Dict[str, float]]

@app.post("/predict")
async def predict(data_input: DataInput):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame(data_input.data)
        # Make predictions
        predictions = make_prediction(data)

        return JSONResponse(content={"predictions": predictions}, status_code=200)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}",
        )
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
