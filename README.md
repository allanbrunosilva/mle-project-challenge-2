
# Sound Realty Price Prediction API - Basic Usage

This FastAPI application serves a machine learning model that predicts house prices based on housing data. It exposes two RESTful endpoints: one for full feature prediction and another for basic input prediction. The model was developed using a structured dataset with additional demographic enrichment by ZIP code.

---

## Features

- **Predict Endpoint**: Uses full feature set including zipcode-merged demographics.
- **Basic Predict Endpoint**: Uses a subset of core features only.
- **Health Check**: Lightweight readiness probe.
- **Model Metadata**: Versioning and feature count are returned in predictions.

---

## Requirements

- Python 3.9+
- `pydantic`
- `pandas`
- `scikit-learn` (for inference)
- `uvicorn`
- `fastapi`

## Environment Setup

The model developer at Sound Realty used a [Conda environment](https://docs.conda.io/en/latest/) to create the model, which has been captured in Conda's YAML format.

Assuming Conda has been installed in your environment, you can recreate the environment with the following:

```sh
conda env create -f conda_environment.yml
# Activate the environment. Repeat this for new terminal sessions
conda activate housing
```

## Train the Model

Once you've created and activated the environment, you can run the script which creates the model:

```sh
python create_model.py
```

This will train the model and save the following artifacts in a directory called `model/`:

- `model/model.pkl` – The trained model serialized in Pickle format.
- `model/model_features.json` – The list of features (and their order) used during training.

## Run the API Locally

With the model artifacts in place, start the API server using FastAPI and Uvicorn:

```sh
uvicorn app:app --reload --port 8000
```

The server will be available at: [http://localhost:8000](http://localhost:8000)

## Available API Endpoints

### Root
```http
GET /
```
Root endpoint that returns a welcome message.

### Health Check
```http
GET /health
```
Health check for verifying the app is live and artifacts are loaded.

### Predict with Full Feature Set
```http
POST /predict
Content-Type: application/json
```
Takes a full feature set and returns a prediction along with model metadata.

**Request body:**
```json
{
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "floors": 1.0,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "sqft_above": 1800,
  "sqft_basement": 0,
  "yr_built": 1995,
  "yr_renovated": 0,
  "zipcode": "98103",
  "lat": 47.65,
  "long": -122.34,
  "sqft_living15": 1600,
  "sqft_lot15": 4000
}
```

### Predict with Basic Features Only
```http
POST /predict_basic
Content-Type: application/json
```
Accepts a subset of basic features for predictions.

**Request body:**
```json
{
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "floors": 1.0,
  "sqft_above": 1800,
  "sqft_basement": 0,
  "zipcode": "98103"
}
```

## Expected Response Format
```json
{
  "prediction": 723456.78,
  "model_version": "0.1.0",
  "features_used": 22,
  "time_ms": 5,
  "warnings": []
}
```
---

For questions or improvements, open an issue or contact the project maintainer.
