import os, sys
import certifi
from dotenv import load_dotenv
import pymongo

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.helper.utils import load_object
from src.utils.ml_utils.model.estimator import NetworkModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, File, UploadFile
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME


ca = certifi.where()
load_dotenv()
templates = Jinja2Templates(directory="./templates")

mongo_db_url = os.getenv("MONGO_DB_URL")
print("MongoDB URL:", mongo_db_url)
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train", tags=["train"])
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response(content="Training successfully completed ✅", media_type="text/plain")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
@app.post("/predict", tags=["predict"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        logging.info("Loaded model and preprocessor successfully.")

        y_pred = network_model.predict(df)
        logging.info("Prediction completed successfully.")

        df['predicted_column'] = y_pred
        logging.info(f"Predictions: {df['predicted_column'].tolist()}")

        df.to_csv("test/predictions/output.csv", index=False)
        
        table_html = df.to_html(
            classes='table table-striped table-bordered table-hover', 
            index=False,
            border=0
        )
        return templates.TemplateResponse(
            "table.html", 
            {"request": request, "table": table_html}
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)