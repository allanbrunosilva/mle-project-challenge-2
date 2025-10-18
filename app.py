import os, pathlib, pickle, json, time
import pandas as pd
from typing import List, Tuple, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# Config section: uses environment variables for flexibility during deployment
# These paths can be overridden by Docker env vars or system environment vars.
MODEL_DIR = os.getenv("MODEL_DIR", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.json")
DEMOGRAPHICS_PATH = os.getenv("DEMOGRAPHICS_PATH", "data/zipcode_demographics.csv")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0") # Typical convention for local dev versions

# Function to safely load all model artifacts before starting the API
def load_artifacts():
    if not pathlib.Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    if not pathlib.Path(FEATURES_PATH).exists():
        raise RuntimeError(f"Features JSON not found at {FEATURES_PATH}")
    if not pathlib.Path(DEMOGRAPHICS_PATH).exists():
        raise RuntimeError(f"Demographics CSV not found at {DEMOGRAPHICS_PATH}")
    
    # Load the serialized model and metadata
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    # Read ZIP codes as strings to preserve leading zeros (important for merge)
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})

    return model, features, demographics

# Load model and data into memory once — shared across requests for performance, which avoids expensive reloads per call
model, model_features, demographics = load_artifacts()

app = FastAPI(
    title="Sound Realty House Price Predictor",
    description="Predict house prices in the Sound Realty dataset.",
    version=APP_VERSION
)

# Schemas
class PredictRequest(BaseModel):
    """Full feature set expected by the model."""
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: str # Define ZIP codes as strings to preserve leading zeros
    lat: float 
    long: float 
    sqft_living15: int
    sqft_lot15: int

class BasicPredictRequest(BaseModel):
    """Subset of key features for a lighter endpoint."""
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: str # Define ZIP codes as strings to preserve leading zeros

class PredictResponse(BaseModel):
    """Standardized response format."""
    prediction: float
    model_version: str
    features_used: int
    time_ms: int # Time in milliseconds
    warnings: Optional[List[str]] = None

# Helper function
def _prepare_frame(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepares the input frame for model inference.
    """
    if input_data.empty:
        raise HTTPException(status_code=400, detail="400 - Bad request: Empty payload")

    # Sanity check: warn if critical inputs are zero
    critical_features = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "sqft_above", "floors", "lat", "long"
    ]
    warnings = []
    for feature in critical_features:
        if feature in input_data.columns:
            zero_mask = input_data[feature] == 0
            if zero_mask.any():
                warnings.append(f"Feature '{feature}' has zero value(s).")

    # Merge input_data with demographics table
    merged_data = input_data.merge(demographics, how="left", on="zipcode")

    # Reorder columns to match training feature ordering
    X = merged_data[model_features]

    return X, warnings

# API Endpoints
@app.get("/")
def root():
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to the Sound Realty House Price Predictor"}

@app.get("/health")
def health():
    """Simple health check endpoint for monitoring."""
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "model_features_count": len(model_features),
        "demographics_rows": int(demographics.shape[0])
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest = Body(...)):
    """Predict house price using the full feature set."""
    t0 = time.perf_counter() # More precise than time.time() for short durations
    request_df = pd.DataFrame([request.dict()]) # Convert the (Pydantic) request to a plain dict and wrap in a list so pandas builds a single-row DataFrame with correct column names. Passing the model object directly makes pandas treat it as a generic object and you end up with numeric columns (leading to KeyError like 'zipcode').
    
    X, warnings = _prepare_frame(request_df)
    try:
        y = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail= f"500 - Internal Server Error: Model inference failed: {e}")
    
    dt_ms = int((time.perf_counter() - t0) * 1000) # Delta time in milliseconds
    return PredictResponse(
        prediction=float(y),
        model_version = APP_VERSION,
        features_used = len(model_features),
        time_ms = dt_ms,
        warnings = warnings or None
    )

@app.post("/predict_basic", response_model=PredictResponse)
async def predict_basic(request: BasicPredictRequest = Body(...)):
    """Predict house price using only basic features."""
    t0 = time.perf_counter()
    request_df = pd.DataFrame([request.dict()])
    X, warnings = _prepare_frame(request_df)

    try:
        y = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail= f"500 - Internal Server Error: Model inference failed: {e}")
    
    dt_ms = int((time.perf_counter() - t0) * 1000) # Delta time in milliseconds
    return PredictResponse(
        prediction=float(y),
        model_version = APP_VERSION,
        features_used = len(model_features),
        time_ms = dt_ms,
        warnings = warnings or None
    )
