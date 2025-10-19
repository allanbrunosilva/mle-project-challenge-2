import os, pathlib, pickle, json, time
import pandas as pd
from typing import List, Tuple, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# Config section: uses environment variables for flexibility during deployment
# These paths can be overridden by Docker env vars or system environment vars.

# App version string
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

# Base model directory (default is "model/")
BASE_MODEL_DIR = os.getenv("MODEL_DIR", "model")

# Path to version marker file
VERSION_FILE = pathlib.Path(BASE_MODEL_DIR) / "version.txt"

# Default fallback version for local dev/testing
DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")
MODEL_VERSION = (
    VERSION_FILE.read_text().strip()
    if VERSION_FILE.exists()
    else DEFAULT_MODEL_VERSION
)

# Initial model state
model = None
model_features = None
demographics = None
last_loaded_version = None

def load_artifacts(model_version: str):
    """Loads model artifacts for a given version."""
    model_dir = os.path.join(BASE_MODEL_DIR, model_version)
    model_path = os.path.join(model_dir, "model.pkl")
    features_path = os.path.join(model_dir, "model_features.json")
    demographics_path = os.getenv("DEMOGRAPHICS_PATH", "data/zipcode_demographics.csv")

    # Validate existence
    if not pathlib.Path(model_path).exists():
        raise RuntimeError(f"Model not found at {model_path}")
    if not pathlib.Path(features_path).exists():
        raise RuntimeError(f"Feature list not found at {features_path}")
    if not pathlib.Path(demographics_path).exists():
        raise RuntimeError(f"Demographics CSV not found at {demographics_path}")

    # Load artifacts
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "r") as f:
        features = json.load(f)
    demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})

    return model, features, demographics

def load_model_if_updated():
    """Reloads the model only if version has changed (based on version.txt)."""
    global model, model_features, demographics, last_loaded_version

    try:
        current_version = VERSION_FILE.read_text().strip()
    except FileNotFoundError:
        current_version = DEFAULT_MODEL_VERSION

    if current_version != last_loaded_version:
        model, model_features, demographics = load_artifacts(current_version)
        last_loaded_version = current_version
        print(f"Model reloaded: {current_version}")


# Load model and data into memory once — shared across requests for performance, which avoids expensive reloads per call
model, model_features, demographics = load_artifacts(MODEL_VERSION)

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
    load_model_if_updated()  # Reloads model if needed
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
    load_model_if_updated()  # Reloads model if needed
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
