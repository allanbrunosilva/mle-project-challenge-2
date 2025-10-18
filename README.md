
# Sound Realty Price Prediction API - Basic Usage

This FastAPI application serves a machine learning model that predicts house prices based on housing data. It exposes two RESTful endpoints: one for full feature prediction and another for basic input prediction. The model was developed using a structured dataset with additional demographic enrichment by ZIP code.

---

## Features

- **Predict Endpoint**: Uses full feature set including zipcode-merged demographics.
- **Basic Predict Endpoint**: Uses a subset of core features only.
- **Health Check**: Lightweight readiness probe.
- **Model Metadata**: Versioning and feature count are returned in predictions.
- **Docker-Compatible**: The API and test script can be used in containerized environments.

---

## Requirements

- Python 3.9+
- `pydantic`
- `pandas`
- `scikit-learn` (for inference)
- `uvicorn`
- `fastapi`
- `requests`

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

---

## Evaluate the Model

After training, the model is evaluated using standard metrics:

* **R² (coefficient of determination)** — how well the model explains variance in prices
* **MAE (mean absolute error)** — average dollar error per prediction
* **RMSE (root mean squared error)** — penalizes larger errors more

Evaluation runs automatically as part of the training script:

```bash
python create_model.py
```

This prints results like:

```
Evaluating model: KNeighborsRegressor

Train Set Performance
  R²:   0.8414 | The model explains 84% of the variance in the target variable
  MAE:  76,233 | On average, the model's predictions are off by about $76,233
  RMSE: 143,467 | Typical prediction error is around $143,467, with larger misses penalized more

Test Set Performance
  R²:   0.7281 | The model explains 73% of the variance in the target variable
  MAE:  102,044 | On average, the model's predictions are off by about $102,044
  RMSE: 201,659 | Typical prediction error is around $201,659, with larger misses penalized more

Potential overfitting: train R² much higher than test R².
```

> Evaluation code can be found in `evaluate_model.py`, which is imported into the training pipeline.

---

### Model Comparison Results

| Model                     | Split |   R² | MAE ($) | RMSE ($) | Notes                         |
| :------------------------ | :---- | ---: | ------: | -------: | :---------------------------- |
| **KNeighborsRegressor**   | Train | 0.84 |  76,232 |  143,467 | Baseline model                |
|                           | Test  | 0.73 | 102,045 |  201,659 | Slight overfitting            |
| **RandomForestRegressor** | Train | 0.97 |  33,013 |   60,326 | Captures nonlinearities well  |
|                           | Test  | 0.78 |  93,767 |  180,591 | Better generalization overall |

### 📈 Interpretation

* The **RandomForestRegressor** achieved higher R² on test data (0.78 vs 0.73) and reduced both MAE and RMSE.
* Training R² (0.97) indicates the model fits data well, though a gap versus test R² (0.78) suggests mild overfitting — acceptable for an 80 % practical solution.
* Random Forests handle nonlinear relationships, feature interactions, and outliers better than KNN, explaining the improvement.

> The chosen RandomForestRegressor provides a strong, interpretable baseline suitable for deployment and future scaling. Note: `random_state` set to 42.

````

---

## CI Integration

Model evaluation runs automatically on every push or pull request to `develop` and `main`.

* A comment with evaluation results is posted directly on pull requests
* CI also verifies that model artifacts were saved (`model.pkl`, `model_features.json`)

> Workflow: `.github/workflows/model_evaluation.yml`

---

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

## Test the API

Use the provided `test_api.py` script to send test requests based on real unseen examples.

```bash
python test_api.py
```

This script:

* Verifies the API is live (`/health`)
* Loads examples from `data/future_unseen_examples.csv`
* Sends 10 test requests to the `/predict` endpoint
* Prints formatted responses

### Override API URL (Docker scenario)

You can override the API base URL using an environment variable:

```bash
API_URL=http://api:8000 python test_api.py
```

This is useful when running in Docker or `docker-compose` setups where services communicate by name.

---

For questions or improvements, open an issue or contact the project maintainer.







