# 🐳 Docker Deployment Guide - Predictive Maintenance AI

This guide explains how to deploy Predictive Maintenance application using Docker and Docker Compose.

## 📋 Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)
- DagsHub account with access token
- Your trained models pushed to DagsHub MLflow Registry with `@production` alias

---

## 🚀 Quick Start

### 1. **Prepare Environment Variables**

Create .env file with following dagshub credentials
```
REPO_OWNER=your-dagshub-username
REPO_NAME=your-repo-name
DAGSHUB_TOKEN=your-dagshub-personal-access-token
API_URL=http://fastapi:8000
```

**⚠️ Important:** Never commit `.env` to git! It's already in `.gitignore`.

---

### 2. **Build and Run**

From your project root directory:

```bash
# Build images 
docker-compose build

# Start services
docker-compose up
```

**Or run in detached mode (background):**

```bash
docker-compose up -d
```

---

### 4. **Access the Application**

- **Streamlit UI:** http://localhost:8501
- **FastAPI Docs:** http://localhost:8000/docs
- **API Health:** http://localhost:8000/health

---

## 🔧 Common Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f fastapi
docker-compose logs -f streamlit
```

### Stop Services

```bash
# Stop containers (keeps images and volumes)
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove everything including volumes
docker-compose down -v
```

### Rebuild After Code Changes

```bash
# Rebuild and restart
docker-compose up --build

# Rebuild specific service
docker-compose build fastapi
docker-compose up -d fastapi
```

### Execute Commands Inside Containers

```bash
# Access FastAPI container shell
docker exec -it predictive_maintenance_api bash

# Access Streamlit container shell
docker exec -it predictive_maintenance_ui bash

# Run Python in container
docker exec -it predictive_maintenance_api python -c "import mlflow; print(mlflow.__version__)"
```

---

## 📂 Project Structure for Docker

```
predictive-maintenance/
├── Dockerfile.fastapi          # FastAPI container definition
├── Dockerfile.streamlit        # Streamlit container definition
├── docker-compose.yml          # Orchestration configuration
├── .dockerignore              # Files to exclude from build
├── .env                       # Your secrets (DON'T COMMIT!)
├── .env.example               # Template for .env
├── requirements.txt           # Python dependencies
├── params.yaml                # Model parameters
└── src/
    ├── api/
    │   └── app.py            # FastAPI application
    ├── ui/
    │   └── streamlit_ui.py   # Streamlit dashboard
    ├── models/
    │   ├── predict.py        # Model manager & inference
    │   └── shap_analysis.py  # SHAP explanations
    └── features/
        └── feature_engineering.py
```

---

## 🔍 How It Works

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
│                                                              │
│  ┌─────────────────────┐      ┌────────────────────┐        │
│  │  FastAPI Container  │      │ Streamlit Container│        │
│  │  • Python 3.12.7    │◄─────┤ • Python 3.12.7    │        │
│  │  • Port: 8000       │      │ • Port: 8501       │        │
│  │  • Loads models     │      │ • Calls FastAPI    │        │
│  │  • Serves /predict  │      │ • UI Dashboard     │        │
│  └──────────┬──────────┘      └────────────────────┘        │
│             │                                                │
│             │ Fetches @production models                     │
│             │                                                │
│  ┌──────────▼──────────────────────────────────────┐        │
│  │         Persistent Volumes                      │        │
│  │  • ./models/ (bind mount - visible on host)    │        │
│  │  • mlflow_cache (Docker volume - hidden)       │        │
│  └─────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  DagsHub MLflow     │
    │  • LGBM_Model       │
    │  • RF_Model         │
    │  • Thresholds       │
    └─────────────────────┘
```

---

## 🎯 Key Features

### ✅ Model Management
- **Automatic loading** from DagsHub on startup
- **Persistent caching** via Docker volumes (models downloaded once!)
- **100-minute TTL** in-memory cache
- **Aliases** for version management (`@production`)

### ✅ Persistent Storage
- **./models/ folder** - Bind mount to host (visible in your project)
- **mlflow_cache volume** - Docker-managed volume for MLflow artifacts
- **No re-downloads** on container restarts

### ✅ Service Communication
- **Internal network** for FastAPI ↔ Streamlit
- **Health checks** ensure FastAPI is ready before Streamlit starts
- **Auto-restart** on failures

### ✅ Environment Flexibility
- **Local development:** Uses `http://127.0.0.1:8000`
- **Docker deployment:** Uses `http://fastapi:8000`
- **No code changes** needed between environments

---

## 📊 What Happens on Startup

### FastAPI Container:
1. ✅ Installs dependencies from `requirements.txt`
2. ✅ Loads environment variables (REPO_OWNER, REPO_NAME, DAGSHUB_TOKEN)
3. ✅ Connects to DagsHub MLflow
4. ✅ Fetches `LGBM_Model@production` and `RF_Model@production`
5. ✅ Downloads threshold artifacts
6. ✅ Starts Uvicorn server on port 8000
7. ✅ Health check endpoint becomes available

### Streamlit Container:
1. ✅ Waits for FastAPI health check to pass
2. ✅ Installs dependencies
3. ✅ Loads environment variables
4. ✅ Connects to FastAPI via `http://fastapi:8000`
5. ✅ Starts Streamlit server on port 8501

---

## 🔧 Common Operations

### View Logs
```bash
# All services
docker-compose logs -f

# Only FastAPI
docker-compose logs -f fastapi

# Only Streamlit
docker-compose logs -f streamlit
```

### Stop Services
```bash
docker-compose down
```

### Restart After Code Changes
```bash
docker-compose up --build
```

### Access Container Shell
```bash
# FastAPI
docker exec -it predictive_maintenance_api bash

# Streamlit
docker exec -it predictive_maintenance_ui bash
```

---

## 🛡️ Security Checklist

- [ ] `.env` is in `.gitignore`
- [ ] `.env` is NOT committed to git
- [ ] DagsHub token has minimal permissions
- [ ] Using read-only token if possible
- [ ] Secrets are not hardcoded in Dockerfiles
- [ ] Production uses external secrets management

---

## 🛠️ Troubleshooting

### Problem: "Could not sync with Registry API"

**Cause:** FastAPI container not running or not healthy

**Solution:**
```bash
# Check container status
docker-compose ps
# View FastAPI logs
docker-compose logs fastapi
# Restart FastAPI
docker-compose restart fastapi
```

---

### Problem: "Critical Error loading @production"

**Cause:** Invalid DagsHub credentials or model not found

**Solution:**
1. Verify `.env` credentials are correct
2. Check if models exist in DagsHub MLflow with `@production` alias
3. Test authentication:

```bash
docker exec -it predictive_maintenance_api python -c "
import os
import dagshub
dagshub.init(
    repo_name=os.getenv('REPO_NAME'),
    repo_owner=os.getenv('REPO_OWNER'),
    mlflow=True
)
import mlflow
print('MLflow URI:', mlflow.get_tracking_uri())
"
```

---

### Problem: Models not updating

**Cause:** Cache TTL hasn't expired (100 min)

**Solution:**
```bash
# Restart containers to force model reload
docker-compose restart
```

---

### Problem: Port already in use

**Cause:** Another service using port 8000 or 8501

**Solution:**
Edit `docker-compose.yml` to use different ports:

```yaml
services:
  fastapi:
    ports:
      - "8080:8000"  # Change host port
  streamlit:
    ports:
      - "8502:8501"  # Change host port
```

Then update `.env`:
```bash
API_URL=http://fastapi:8000  # Keep this (internal)
```

---

### Problem: Import errors in containers

**Cause:** Missing dependencies in `requirements.txt`

**Solution:**
1. Update `requirements.txt`
2. Rebuild images:

```bash
docker-compose build --no-cache
docker-compose up
```

---

## 🔒 Security Best Practices

1. **Never commit `.env`** - Already in `.gitignore`
2. **Rotate DagsHub tokens** periodically
3. **Use secrets management** in production (e.g., Docker Secrets, AWS Secrets Manager)
4. **Limit token permissions** - Use read-only tokens if possible
5. **Network isolation** - In production, don't expose FastAPI port externally

---

## 🚢 Production Deployment

For production environments, consider:

### 1. Use Environment-Specific Compose Files

```bash
# docker-compose.prod.yml
version: '3.8'
services:
  fastapi:
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

Run with:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### 2. Use External Secrets

Don't use `.env` file - inject secrets at runtime:

```bash
docker-compose run -e DAGSHUB_TOKEN=$DAGSHUB_TOKEN fastapi
```

### 3. Add Reverse Proxy

Use Nginx or Traefik for SSL termination and load balancing.

### 4. Health Monitoring

Set up monitoring with Prometheus + Grafana or use cloud-native solutions.

---

## 📊 Resource Usage

**Typical resource consumption:**

| Service   | CPU  | Memory | Storage |
|-----------|------|--------|---------|
| FastAPI   | 0.5  | 1-2 GB | ~500 MB |
| Streamlit | 0.3  | 1 GB   | ~500 MB |

**Note:** Model loading spikes memory temporarily on startup.

---

## 🔄 CI/CD Integration

Add to your GitHub Actions:

```yaml
# .github/workflows/deploy.yml
name: Deploy Docker

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build and Push
        run: |
          docker build -f Dockerfile.fastapi -t myrepo/maintenance-api .
          docker build -f Dockerfile.streamlit -t myrepo/maintenance-ui .
          docker push myrepo/maintenance-api
          docker push myrepo/maintenance-ui
```

---

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [DagsHub MLflow Guide](https://dagshub.com/docs/integration_guide/mlflow/)
- [FastAPI Docker Deployment](https://fastapi.tiangolo.com/deployment/docker/)
- [Streamlit Docker Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)

---

## 🐛 Getting Help

If you encounter issues:

1. Check container logs: `docker-compose logs -f`
2. Verify environment variables: `docker-compose config`
3. Test model loading manually inside container
4. Review DagsHub MLflow Registry

---

## 📝 Changelog

- **v1.0.0** - Initial Docker setup with FastAPI + Streamlit
  - Python 3.12.7 slim base image
  - DagsHub MLflow integration
  - Health checks and auto-restart
  - Configurable API URL for multi-environment support

---

**Happy Deploying! 🚀**
