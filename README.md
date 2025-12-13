
# 🫁 Chest Cancer Classification using Deep Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECS-yellow.svg)](https://aws.amazon.com/ecs/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)

</div>

## 📋 Overview

An end-to-end deep learning solution for detecting **Adenocarcinoma** cancer from chest CT scan images. Built with production-grade MLOps practices, this project demonstrates complete ML pipeline implementation from data ingestion to deployment with automated CI/CD workflows.

## ✨ Key Features

### Machine Learning Pipeline
- **Transfer Learning** with EfficientNetB0 for optimal performance
- **Automated training pipeline** with modular component architecture
- **MLflow integration** for experiment tracking and model versioning
- **DVC (Data Version Control)** for reproducible data pipelines

### Production-Ready Application
- **FastAPI REST API** with clean, async endpoints
- **Interactive web interface** with drag-and-drop image upload
- **Model caching** for sub-second inference after initial load
- **Health check endpoints** for monitoring

### MLOps & DevOps
- **CI/CD Pipeline** with GitHub Actions
- **Docker containerization** with optimized image size
- **AWS ECS deployment** ready with automated workflows
- **Environment-based configuration** for secure credential management

## 🛠️ Tech Stack

### Core ML/DL
- **TensorFlow/Keras** - Deep learning framework
- **EfficientNetB0** - Pre-trained CNN model
- **NumPy, Pandas** - Data manipulation

### MLOps Tools
- **MLflow** - Experiment tracking and model registry
- **DVC** - Data and model versioning
- **DagHub** - Remote experiment tracking

### Backend & API
- **FastAPI** - Modern web framework for building APIs
- **Uvicorn** - ASGI server
- **Python-multipart** - File upload handling

### Frontend
- **TailwindCSS** - Responsive UI design
- **Vanilla JavaScript** - Interactive web interface

### DevOps & Cloud
- **Docker** - Containerization
- **GitHub Actions** - CI/CD automation
- **AWS ECS** - Container orchestration
- **AWS ECR** - Container registry

### Development Tools
- **Python-dotenv** - Environment variable management
- **PyYAML** - Configuration file parsing
- **Python-box** - Dict to object conversion

---📁 Project Structure

```
├── .github/
│   └── workflows/
│       └── main.yaml              # CI/CD pipeline configuration
├── artifacts/
│   ├── data_ingestion/            # Downloaded and processed data
│   ├── prepare_base_model/        # Base and updated models
│   └── training/                  # Trained models and logs
├── config/
│   └── config.yaml                # Project configuration
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_trainer.ipynb
│   └── 04_model_evaluation_with_mlflow.ipynb
├── src/cnnClassifier/
│   ├── components/                # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation_mlflow.py
│   ├── config/
│   │   └── configuration.py       # Configuration manager
│   ├── entity/
│   │   └── config_entity.py       # Configuration dataclasses
│   ├── pipeline/                  # Training and prediction pipelines
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_trainer.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── prediction.py
│   ├── utils/
│   │   └── common.py              # Utility functions
│   └── constants/
│       └── __init__.py            # Project constants
├── templates/
│   └── index.html                 # Web interface
├── app.py                         # FastAPI application
├── main.py                        # Training pipeline entry point
├── dvc.yaml                       # DVC pipeline configuration
├── params.yaml                    # Model hyperparameters
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
├── .dockerignore                  # Docker build exclusions
└── README.md
```

<div align="center">
**⭐ Star this repo if you find it useful**
## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Docker (optional)
- AWS CLI (for deployment)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Chest-Cancer-Classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Run the application**
   ```bash
   python app.py
   ```
   Access at: `http://localhost:8000`

### Docker Deployment

```bash
# Build image
docker build -t chest-cancer-classifier .

# Run container
docker run -p 8000:8000 --env-file .env chest-cancer-classifier
```

## 📊 Model Training Pipeline

```bash
# Run complete training pipeline
python main.py

# Or run individual stages with DVC
dvc repro
```

## 🔗 API Endpoints

- `GET /` - Web interface
- `POST /predict` - Image classification endpoint
- `GET /health` - Health check

## 📈 Results

- **Model**: EfficientNetB0 (Transfer Learning)
- **Input Size**: 224x224x3
- **Classes**: Adenocarcinoma Cancer, Normal
- **Metrics**: Accuracy, Precision, Recall, AUC

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ by [Harsh Pratap Singh](https://github.com/CodeBy-HP)
