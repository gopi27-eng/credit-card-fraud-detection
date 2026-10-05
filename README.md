# credit-card-fraud-detection
# End-to-End Production MLOps: Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning Operations (MLOps) platform built from scratch. This repository covers the complete machine learning lifecycle: data versioning, automated preprocessing, imbalance-aware model training, experiment tracking, containerization, automated testing (CI), continuous training (CT), GitOps continuous deployment (CD), and cluster observability.

---

## 1. System Architecture

```text
                                 +-----------------------+
                                 | Kaggle / S3 Data Sync |
                                 +-----------+-----------+
                                             |
                                        [dvc pull]
                                             v
+------------------+    Git Push         +------------------------+
|  Local Developer | ------------------> |  GitHub Repository     |
+------------------+                     +-----------+------------+
                                                     |
                         +---------------------------+---------------------------+
                         |                                                       |
                         v (Triggers on code change)                             v (Triggers on schedule / data change)
           +---------------------------+                           +---------------------------+
           | GitHub Actions CI Pipeline|                           | GitHub Actions CT Pipeline|
           +-------------+-------------+                           +-------------+-------------+
                         |                                                       |
                         | (Pytest 40+ tests & coverage)                         | (dvc pull data from S3)
                         v                                                       v
           +---------------------------+                           +---------------------------+
           | Docker Buildx (Build)     |                           | Train XGBoost Model       |
           +-------------+-------------+                           +-------------+-------------+
                         |                                                       |
                         v                                                       v (Logs metrics/artifacts)
           +---------------------------+                           +---------------------------+
           | GitHub Packages (GHCR)    |                           | DagsHub MLflow Registry   |
           +-------------+-------------+                           +-------------+-------------+
                         |                                                       |
                         |                                                       v (Pushes updated weights)
                         |                                         +---------------------------+
                         |                                         | Git Auto-Commit (Artifact)|
                         |                                         +---------------------------+
                         |
                         v (Image Reference Update)
           +---------------------------+
           | Argo CD (GitOps Engine)   |
           +-------------+-------------+
                         |
                         | (Declarative Reconcile)
                         v
           +----------------------------------------------------+
           |             Kind Kubernetes Cluster                |
           |                                                    |
           |  +------------------------+                        |
           |  | FastAPI Pods (HPA)     |                        |
           |  | Port: 8000             |                        |
           |  +-----------+------------+                        |
           |              |                                     |
           |              v (Scrapes /metrics)                  |
           |  +------------------------+   Metrics Feed         |
           |  | Prometheus Agent       | ----------------+      |
           |  +------------------------+                 |      |
           |                                             v      |
           |                               +------------------+ |
           |                               | Grafana Dashboard| |
           |                               +------------------+ |
           +----------------------------------------------------+

```

---

## 2. Core Tech Stack

* **Machine Learning & Core:** Python 3.12, XGBoost, Scikit-learn, Pandas, NumPy, Joblib
* **API Service:** FastAPI, Uvicorn, Pydantic
* **Data Versioning:** DVC (Data Version Control) with AWS S3 remote storage
* **Experiment Tracking:** MLflow remote hosted on DagsHub
* **Containerization:** Docker with multi-stage build caching
* **CI/CD/CT Automation:** GitHub Actions, Docker Buildx, GHCR (GitHub Container Registry)
* **Local Cluster Infrastructure:** Kind (Kubernetes in Docker), Kubectl
* **Package Management:** Helm 3
* **GitOps Continuous Deployment:** Argo CD
* **Observability & Monitoring:** Prometheus Community Stack, Grafana

---

## 3. Project Directory Structure

```text
credit-card-fraud-detection/
├── .dvc/                   # DVC internal metadata & config
├── .github/
│   └── workflows/
│       ├── ci.yaml         # Linting, testing (pytest), and GHCR build/push
│       └── ct.yaml         # Data pull, retrain, MLflow logging, model commit
├── artifacts/              # Serialized pipeline binaries (.joblib)
│   ├── model.joblib
│   └── preprocessor.joblib
├── config/                 # Central configuration files
│   └── config.yaml
├── data/                   # Data directory (managed by DVC, ignored by git)
│   ├── raw/
│   └── processed/
│       ├── train.parquet.dvc
│       └── test.parquet.dvc
├── infra/                  # Kubernetes manifests & Helm values
│   ├── argo-app.yaml
│   ├── deployment.yaml
│   ├── service.yaml
|   └── servicemonitor.yaml
├── src/
│   ├── api/
│   │   └── app.py          # FastAPI prediction API with /predict & /metrics
│   ├── data/
│   │   ├── ingest.py       # Data extraction and train/test splitting
│   │   └── transformation.py # Imbalance-aware preprocessing pipeline
│   ├── models/
│   │   ├── predict.py      # Batch/real-time inference wrapper
│   │   └── train.py        # XGBoost training with scale_pos_weight & MLflow
│   ├── schemas/
│   │   └── transaction.py  # Pydantic request/response data contracts
│   └── utils/
│       ├── config.py       # YAML parser and configuration dataclass
│       └── logger.py       # Structured JSON logging utility
├── tests/                  # 41-case comprehensive test suite
├── Dockerfile              # Production multi-stage image definition
├── requirements.txt        # Pinned runtime dependencies
└── README.md

```

---

## 4. Step-by-Step Implementation Guide

### Phase 1: Local Development Environment Setup

Clone the repository and set up a clean Python 3.12 virtual environment:

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-cov dvc[s3] dagshub

```

---

### Phase 2: Data Versioning (DVC + AWS S3)

To ensure datasets do not bloat Git histories while keeping track of data changes:

1. **Initialize DVC:**
```bash
dvc init

```


2. **Configure AWS S3 Remote Storage:**
```bash
dvc remote add -d origin s3://<your-s3-bucket-name>/dvcstore
dvc remote modify origin region us-east-1

```


3. **Track and Push Processed Datasets:**
```bash
# Run ingestion to produce train and test splits
python -m src.data.ingest

# Track datasets with DVC
dvc add data/processed/train.parquet data/processed/test.parquet

# Push to S3 remote storage
dvc push data/processed/train.parquet.dvc data/processed/test.parquet.dvc

# Track the DVC metadata files in Git
git add data/processed/*.dvc data/processed/.gitignore .dvc/config
git commit -m "chore(dvc): track parquet datasets via S3 remote"

```



---

### Phase 3: Experiment Tracking (DagsHub MLflow)

We track runs, parameters, metrics (PR-AUC, ROC-AUC, Recall), and model binaries to a hosted DagsHub instance without needing a dedicated tracking server.

1. Set your credentials in your local terminal:
```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<repo-owner>/credit-card-fraud-detection.mlflow"
export MLFLOW_TRACKING_USERNAME="<your-dagshub-username>"
export MLFLOW_TRACKING_PASSWORD="<your-dagshub-token>"

```


2. Run transformation and training:
```bash
python -m src.data.transformation
python -m src.models.train

```



XGBoost handles the severe class imbalance dynamically via `scale_pos_weight = (neg_count / pos_count)`. All artifacts save to `artifacts/model.joblib` and log directly into DagsHub MLflow.

---

### Phase 4: CI/CD/CT Automation via GitHub Actions

Configure repository secrets in **GitHub $\rightarrow$ Settings $\rightarrow$ Secrets and variables $\rightarrow$ Actions**:

* `AWS_ACCESS_KEY_ID`: IAM Access Key with S3 permissions
* `AWS_SECRET_ACCESS_KEY`: IAM Secret Key
* `MLFLOW_TRACKING_URI`: DagsHub tracking URL
* `MLFLOW_TRACKING_USERNAME`: DagsHub username
* `MLFLOW_TRACKING_PASSWORD`: DagsHub Personal Access Token (PAT)

#### Continuous Integration (`.github/workflows/ci.yaml`)

* **Trigger:** Push to `main`, PR, or manual dispatch.
* **Job 1 (Test):** Sets up Python 3.12, installs dependencies, and runs the 41-case test suite (`pytest tests/ -v --cov=src`).
* **Job 2 (Build & Push):** Logs into GitHub Container Registry (GHCR) using the native `GITHUB_TOKEN` and pushes `ghcr.io/<owner>/credit-card-fraud-detection:latest`.

#### Continuous Training (`.github/workflows/ct.yaml`)

* **Trigger:** Weekly cron schedule, manual trigger, or commits touching `data/**.dvc`.
* Pulls data versions from AWS S3 (`dvc pull`).
* Runs `src.data.transformation` and `src.models.train`.
* Validates performance thresholds against baseline metrics.
* Commits updated `artifacts/` back to the repository using a bot identity, appending `[skip ci]` to prevent recursive pipeline loops.

---

### Phase 5: Local Kubernetes Cluster Setup (Kind)

To simulate an authentic enterprise Kubernetes production environment locally, we run a multi-node cluster using Kind (Kubernetes in Docker).

#### 1. Install Kind & Kubectl

```bash
# Install Kind binary (Linux / WSL2)
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.22.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Install Kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

```

#### 2. Create Cluster Config (`kind-config.yaml`)

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: 8000
        protocol: TCP
  - role: worker
  - role: worker

```

#### 3. Spin Up the Cluster

```bash
kind create cluster --name fraud-detection-cluster --config kind-config.yaml
kubectl cluster-info
kubectl get nodes

```

---

### Phase 6: GitOps Continuous Delivery (Argo CD)

Argo CD handles declarative cluster state synchronization directly from this Git repo.

#### 1. Install Argo CD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for Argo CD pods to become ready
kubectl wait --for=condition=ready pod --all -n argocd --timeout=300s

```

#### 2. Access Argo CD Dashboard

Port-forward the Argo CD server locally:

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443

```

* **URL:** `http://localhost:8080`
* **Username:** `admin`
* **Get Initial Password:**
```bash
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo

```



#### 3. Deploy the GitOps Application

Apply the Argo CD Application manifest to track the `infra/` manifests in Git:

```bash
kubectl apply -f infra/argo-app.yaml

```

Argo CD will automatically detect changes pushed to the `infra/` folder in Git and reconcile the running state in your Kind cluster without manual intervention.

---

### Phase 7: Observability (Prometheus & Grafana via Helm)

#### 1. Install Helm

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

```

#### 2. Deploy Prometheus Community Stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus and Grafana stack into the monitoring namespace
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

```

#### 3. Access Prometheus and Grafana Dashboards

```bash
# Port-forward Grafana
kubectl port-forward svc/monitoring-grafana -n monitoring 3000:80

# Retrieve default admin password for Grafana
kubectl get secret --namespace monitoring monitoring-grafana -o jsonpath="{.data.admin-password}" | base64 -d; echo

```

* Access Grafana at `http://localhost:3000` with user `admin` and the retrieved password.
* Add your ServiceMonitor targeting the `/metrics` endpoint of the fraud detection deployment to visualize prediction throughput, inference latencies, and drift signals.

---

## 5. Verification & Testing

### Running the API Service Locally

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

```

### Sending a Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Time": 406.0,
       "V1": -2.31, "V2": 1.95, "V3": -1.60, "V4": 3.99,
       "V5": -0.52, "V6": -1.42, "V7": -2.53, "V8": 1.39,
       "V9": -2.77, "V10": -2.77, "V11": 3.20, "V12": -2.89,
       "V13": -0.59, "V14": -4.28, "V15": 0.38, "V16": -1.14,
       "V17": -2.83, "V18": -0.01, "V19": 0.41, "V20": 0.12,
       "V21": 0.51, "V22": -0.03, "V23": -0.46, "V24": 0.32,
       "V25": 0.04, "V26": 0.17, "V27": 0.26, "V28": -0.14,
       "Amount": 0.0
     }'

```

**Expected Response:**

```json
{
  "is_fraud": 1,
  "fraud_probability": 0.9842,
  "model_version": "xgboost-v1.0"
}

```

---

## 6. Engineering Decisions & Lessons Learned

* **Python 3.12 Runner Architecture:** Upgraded CI/CT environments to Python 3.12 to align with updated pre-compiled scientific wheels (XGBoost 3.x, Scikit-learn), eliminating runner compilation overhead during package installations.
* **Class Imbalance Strategy:** Standard accuracy metrics fail on fraud detection due to extreme imbalance (0.17% positive cases). The pipeline uses PR-AUC as the primary gate metric and calculates `scale_pos_weight` dynamically during ingestion to weight rare positive classes appropriately.
* **Explicit DVC Target Resolution:** Automated runners running non-interactive DVC require explicit target paths (`data/processed/*.dvc`) and direct AWS credentials configured before execution to ensure data syncing occurs without requiring local cache setups.
* **Non-Interactive CI Authentication:** Handled DagsHub authentication by setting environment variables (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`) to bypass the interactive OAuth prompt required by headless runners.
* **Separation of Concerns:** The GitOps pattern allows model weights to be updated in Git via CT, prompting Argo CD to deploy updated pods across the Kind cluster without developer intervention.
