# 🫁 Chest Cancer Classification using Deep Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECS-yellow.svg)](https://aws.amazon.com/ecs/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)

</div>

---

## 📌 Overview

An **end-to-end deep learning project** for detecting **Adenocarcinoma cancer** from chest CT scan images. The project is designed with **production-grade MLOps practices**, covering everything from data ingestion and training to deployment with automated CI/CD pipelines.

---

## ✨ Key Features

### 🧠 Machine Learning

* Transfer Learning using **EfficientNetB0**
* Modular, reusable **training pipeline**
* **MLflow** for experiment tracking & model versioning
* **DVC** for reproducible data and pipeline management

### 🚀 Production Application

* **FastAPI**-based REST API
* Simple **web UI** with image upload support
* Model **lazy loading & caching** for fast inference
* Health-check endpoint for monitoring

### ⚙️ MLOps & DevOps

* **CI/CD pipelines** using GitHub Actions
* **Dockerized** application for consistent deployment
* **AWS ECS** ready deployment workflow
* Environment-based configuration for secrets

---

## 🛠️ Tech Stack

### Core ML / DL

* TensorFlow & Keras
* EfficientNetB0
* NumPy, Pandas

### MLOps

* MLflow
* DVC
* DagHub

### Backend & API

* FastAPI
* Uvicorn
* Python-multipart

### Frontend

* HTML + TailwindCSS
* Vanilla JavaScript

### DevOps & Cloud

* Docker
* GitHub Actions
* AWS ECS & ECR

---

## 📁 Project Structure

```
├── .github/
│   └── workflows/
│       └── main.yaml              # CI/CD pipeline
├── artifacts/
│   ├── data_ingestion/            # Raw & processed data
│   ├── prepare_base_model/        # Base & updated models
│   └── training/                  # Trained models & logs
├── config/
│   └── config.yaml                # Central configuration
├── research/                      # Jupyter notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_trainer.ipynb
│   └── 04_model_evaluation_mlflow.ipynb
├── src/cnnClassifier/
│   ├── components/                # Core ML components
│   ├── pipeline/                  # Training & inference pipelines
│   ├── config/                    # Configuration manager
│   ├── entity/                    # Dataclasses
│   ├── utils/                     # Utility helpers
│   └── constants/
├── templates/
│   └── index.html                 # Web UI
├── app.py                         # FastAPI app
├── main.py                        # Training pipeline entry
├── dvc.yaml                       # DVC pipeline
├── params.yaml                    # Model parameters
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Docker (optional)
* AWS CLI (for cloud deployment)

### Local Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/CodeBy-HP/Chest-Cancer-Classification.git
   cd Chest-Cancer-Classification
   ```

2. **Create & activate virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   # venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   # update credentials inside .env
   ```

5. **Run the application**

   ```bash
   python app.py
   ```

   Visit: [http://localhost:8000](http://localhost:8000)

---

## 🐳 Docker Setup

```bash
# Build image
docker build -t chest-cancer-classifier .

# Run container
docker run -p 8000:8000 --env-file .env chest-cancer-classifier
```

---

## 📊 Training Pipeline

```bash
# Run full training pipeline
python main.py

# Or via DVC
dvc repro
```

---

<div align="center">

⭐ **Star this repository if you find it useful** ⭐

</div>
