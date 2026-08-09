import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from ml.data import process_data
from ml.model import inference, load_artifacts

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

app = FastAPI(
    title="Census Income Classifier API",
    description="Predicts whether a person's income exceeds $50K/year "
    "based on US Census data.",
    version="1.0.0",
)

model, encoder, lb = load_artifacts()


class CensusRecord(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get("/")
async def welcome():
    return {"message": "Welcome to the census income classifier API."}


@app.post("/predict")
async def predict(record: CensusRecord):
    row = pd.DataFrame([record.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        row,
        categorical_features=CATEGORICAL_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    pred = inference(model, X)
    salary = lb.inverse_transform(pred)[0]
    return {"salary": salary}
