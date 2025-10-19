
# Sound Realty Price Prediction API - Basic Usage

This FastAPI application serves a machine learning model that predicts house prices based on housing data. It exposes two RESTful endpoints: one for full feature prediction and another for basic input prediction. The model was developed using a structured dataset with additional demographic enrichment by ZIP code.

---

## Features

- **Predict Endpoint**: Uses full feature set including zipcode-merged demographics.
- **Basic Predict Endpoint**: Uses a subset of core features only.
- **Health Check**: Lightweight readiness probe.
- **Model Metadata**: Versioning and feature count are returned in predictions.
- **Dynamic Model Versioning**: Automatically reloads model if updated on disk.
- **Zero-Downtime Reloads**: Predictions continue uninterrupted while model versions update.
- **Docker-Compatible**: The API and test script can be used in containerized environments.
- **Kubernetes-Ready**: Supports autoscaling and version-aware model loading.

---

## Branches & Deliverables

| Branch                                                                                                                      | Purpose                                                                                   | Connected Requirement                                                                                   |
|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [`main`](https://github.com/allanbrunosilva/mle-project-challenge-2/tree/main)                                              | Stable version of the project                                                             | Final project base                                                                                      |
| [`develop`](https://github.com/allanbrunosilva/mle-project-challenge-2/tree/develop)                                        | Integration branch for active development                                                 | Consolidates feature branches                                                                           |
| [`feature/api-endpoint`](https://github.com/allanbrunosilva/mle-project-challenge-2/tree/feature/api-endpoint)              | RESTful API for `/predict` and `/predict_basic`, adds backend enrichment                  | Req 1: "POST JSON, join demographic data, return prediction + metadata", Bonus: "subset of features"    |
| [`feature/test_script`](https://github.com/allanbrunosilva/mle-project-challenge-2/tree/feature/test_script)                | Python script that submits test samples to `/predict` endpoint                            | Req 2: "Submit examples from data/future_unseen_examples.csv"                                           |
| [`feature/model-evaluation`](https://github.com/allanbrunosilva/mle-project-challenge-2/tree/feature/model-evaluation)      | Adds metrics and evaluation reports (R², MAE, RMSE)                                       | Req 3: "Evaluate how well the model generalizes to new data"                                            |
| [`experiment/random-forest`](https://github.com/allanbrunosilva/mle-project-challenge-2/tree/experiment/random-forest)      | Replaces `KNeighborsRegressor` with `RandomForestRegressor` for improved generalization   | Req: "use traditional ML, 80% solution"                                                                 |
| [`deployment/scaling-strategy`](https://github.com/allanbrunosilva/mle-project-challenge-2/tree/deployment/scaling-strategy)| Adds Docker + Kubernetes deployment with autoscaling support                              | Req: "scale API resources without stopping the service", "zero-downtime deployment with new versions"   |

---

## Requirements

- Python 3.9+
- `pydantic`
- `pandas`
- `scikit-learn` (for inference)
- `uvicorn`
- `fastapi`
- `requests`

---

## Environment Setup

The model developer at Sound Realty used a [Conda environment](https://docs.conda.io/en/latest/) to create the model, which has been captured in Conda's YAML format.

Assuming Conda has been installed in your environment, you can recreate the environment with the following:

```sh
conda env create -f conda_environment.yml
# Activate the environment. Repeat this for new terminal sessions
conda activate housing
```

---

## Train and Version the Model

Once you've created and activated the environment, you can run the script which creates the model:

```sh
python create_model.py
```

This will train the model and save the following artifacts in a **versioned subdirectory** of `model/`:

```
model/
└── v1/
    ├── model.pkl                # The trained model serialized in Pickle format
    └── model_features.json     # The list of features (and their order) used during training
```

The active model version is defined in:

```
model/version.txt
```
This file contains a single line like:

```
v1
```

> `version.txt` is automatically overwritten every time a new model is trained.

---

### Switching Model Versions

To deploy a new model **without restarting the service**, simply update `version.txt` with the desired version:

```text
v2
```

The API will detect the change and **reload the model on the next request**, supporting:

* Zero-downtime model updates
* Smooth integration with CI/CD pipelines or model registries

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

## Model Comparison Results

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

---

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

## Kubernetes Autoscaling with Minikube

This section explains how to **scale this FastAPI service automatically** based on CPU usage using **Kubernetes Horizontal Pod Autoscaler (HPA)** and **Dockerized deployment**. Autoscaling ensures this API can handle spikes in traffic by **automatically adding more pods** when CPU usage rises, and **scaling back down** when it's idle — without downtime.

### Prerequisites

Make sure you have:

* [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
* Windows PowerShell or terminal available
* Admin access (for installing tools)

### Install Required Tools (Windows via Chocolatey)

```powershell
choco install kubernetes-cli
choco install minikube
choco install kind
```

### Start Kubernetes with Minikube

```powershell
docker desktop start
minikube start
kubectl get nodes
```

You should see:

```
NAME       STATUS   ROLES           AGE     VERSION
minikube   Ready    control-plane   Xs      v1.34.0
```

### Build and Deploy the App

#### 1. Point Docker to Minikube

```powershell
& minikube -p minikube docker-env --shell powershell | Invoke-Expression
```

#### 2. Build the Docker image inside Minikube:

```powershell
docker build -t house-price-api:latest .
```

#### 3. Deploy the app to Kubernetes

```powershell
kubectl apply -f deployment.yaml
kubectl expose deployment house-price-api --type=NodePort --port=8000
kubectl get pods
```

#### 4. Port forward for local access

```powershell
kubectl port-forward deployment/house-price-api 8000:8000
```

Access the app:
[http://localhost:8000/health](http://localhost:8000/health)


### Enable Autoscaling

#### 1. Enable metrics server (required for HPA)

```powershell
minikube addons enable metrics-server
```

#### 2. Apply autoscaling

Choose either:

```powershell
kubectl autoscale deployment house-price-api --cpu=50% --min=1 --max=5
```

Or the YAML file:

```powershell
kubectl apply -f hpa.yaml
```

#### 3. Watch autoscaler status

Choose either:

```powershell
kubectl get hpa -w
```

Or live loop with:

```powershell
while ($true) { kubectl get hpa; Start-Sleep -Seconds 2; Clear-Host }
```

You’ll initially see:

```
NAME              REFERENCE                    TARGETS       MINPODS   MAXPODS   REPLICAS
house-price-api   Deployment/house-price-api   cpu: 2%/50%    1         5         1
```

### Generate Load to Trigger Scaling

```powershell
kubectl run -i --tty load-generator --rm --image=busybox -- /bin/sh
```

Inside the pod shell:

```sh
while true; do wget -q -O- http://house-price-api:8000/health; done
```

You’ll then see the HPA scale up:

```
cpu: 368%/50%   replicas: 1 → 4 → 5
```

### Scale-Down After Load

Stop the generator with `Ctrl+C`.

Watch replicas reduce:

```
cpu: 83%/50%   replicas: 5
cpu: 9%/50%    replicas: 2
cpu: 2%/50%    replicas: 1
```

### Cleanup

Delete the load generator pod (if still running):
```powershell
kubectl delete pod load-generator --ignore-not-found
```

Delete the service exposing the app:
```powershell
kubectl delete service house-price-api --ignore-not-found
```

Delete either HPA (depending on which one you created):
```powershell
kubectl delete hpa house-price-api --ignore-not-found
kubectl delete hpa house-price-api-hpa --ignore-not-found
```

Delete the deployment and its pods:
```powershell
kubectl delete deployment house-price-api --ignore-not-found
```

Stop the Minikube cluster (keeps your setup for later):
```powershell
minikube stop
```

**(Optional)** Completely remove the Minikube cluster if you want a fresh start:
```powershell
minikube delete
```

---

## ☁️ Deploying to AWS (Production Translation Example)

You can transition this architecture to a production-grade environment using **Amazon Web Services (AWS)** with minimal changes:

### Components Mapping

| Local/Dev Setup      | Production Equivalent on AWS                                                                |
| -------------------- | ------------------------------------------------------------------------------------------- |
| Docker Desktop       | Amazon ECR (Elastic Container Registry) for storing Docker images                           |
| Minikube (local k8s) | Amazon EKS (Elastic Kubernetes Service) – managed Kubernetes                                |
| `kubectl` CLI        | Still used in AWS (same commands!)                                                          |
| Load Generator       | AWS CloudWatch Metrics or Locust/Artillery for load testing                                 |
| `kubectl autoscale`  | Kubernetes Horizontal Pod Autoscaler (HPA) running in EKS with Cluster Autoscaler for nodes |
| Local port-forward   | AWS ALB (Application Load Balancer) or NLB (Network Load Balancer)                          |
| Local storage        | AWS EBS or S3 for model/data artifacts                                                      |

### Steps to Deploy on AWS

#### 1. **Container Registry**

* Push your Docker image to Amazon ECR:

  ```bash
  aws ecr create-repository --repository-name house-price-api
  docker tag house-price-api:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/house-price-api
  docker push <your-ecr-url>
  ```

#### 2. **Kubernetes Cluster**

* Use **Amazon EKS** (Elastic Kubernetes Service) to create a managed Kubernetes cluster.
* Either via:

  * AWS Console wizard (UI)
  * Or CLI with `eksctl`:

    ```bash
    eksctl create cluster --name house-price-cluster --region us-west-2 --nodes 2 --node-type t3.medium
    ```

#### 3. **Model & Data Storage**

* Store model artifacts (`model.pkl`, `model_features.json`, etc.) in:

  * Amazon S3, and mount/download at container start
  * Or build them into the image (as you currently do)

#### 4. **Scaling**

* Enable **Horizontal Pod Autoscaler (HPA)**, just like in Minikube
* Also enable **Cluster Autoscaler** to add/remove EC2 worker nodes as needed
* Monitor scaling with **CloudWatch**

#### 5. **Public Access**

* Use a **LoadBalancer** service in Kubernetes:

  ```yaml
  type: LoadBalancer
  ```

  This will provision an AWS ALB/NLB automatically and expose your app publicly.

#### 6. **Security & Monitoring**

* Use AWS IAM roles for access control
* Use **CloudWatch** for logs, metrics, and health alerts
* Optionally add **AWS WAF** or **API Gateway** for extra security

### Conclusion

This setup allows **dynamic model versioning**, **horizontal scaling**, and **zero-downtime updates** — ready for real-world cloud deployment.

---

If you have any suggestions, questions or improvements for this project, please **give us feedback** — open an issue or contact the project maintainers.
