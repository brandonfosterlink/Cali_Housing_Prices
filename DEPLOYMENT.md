# 🚀 GCP Deployment Guide - Housing Price Predictor

This guide walks you through deploying your Housing Price Predictor model on Google Cloud Platform using Docker and Terraform.

## 📋 Prerequisites

Before deploying, ensure you have:

1. **Google Cloud SDK** installed and configured
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   
   # Authenticate
   gcloud auth login
   gcloud auth application-default login
   ```

2. **Docker** installed
   ```bash
   # Install Docker Desktop or Docker Engine
   # Visit: https://docs.docker.com/get-docker/
   ```

3. **Terraform** installed
   ```bash
   # Install Terraform
   # Visit: https://learn.hashicorp.com/tutorials/terraform/install-cli
   ```

## 🏗️ Architecture

Your deployment will include:

- **Cloud Run**: Serverless container platform for your Flask app
- **Artifact Registry**: Secure Docker image storage
- **Cloud Build**: Automated CI/CD pipeline (optional)
- **IAM**: Proper permissions and security

## 🚀 Quick Deployment

### Option 1: Automated Deployment (Recommended)

```bash
# Make the script executable (if not already)
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

This script will:
1. ✅ Verify prerequisites
2. 🔧 Set up GCP project and APIs
3. 🏗️ Deploy infrastructure with Terraform
4. 🐳 Build and push Docker image
5. 🚀 Deploy to Cloud Run
6. 📱 Provide service URLs

### Option 2: Manual Deployment

#### Step 1: Set up GCP Project
```bash
# Set your project
gcloud config set project logical-vim-472815-h7

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

#### Step 2: Deploy Infrastructure
```bash
# Navigate to terraform directory
cd terraform

# Initialize Terraform
terraform init

# Review the plan
terraform plan -var-file="terraform.tfvars"

# Apply the infrastructure
terraform apply -var-file="terraform.tfvars"
```

#### Step 3: Build and Deploy Application
```bash
# Go back to project root
cd ..

# Configure Docker for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build Docker image
docker build -t us-central1-docker.pkg.dev/logical-vim-472815-h7/housing-price-predictor/housing-price-predictor:latest .

# Push to registry
docker push us-central1-docker.pkg.dev/logical-vim-472815-h7/housing-price-predictor/housing-price-predictor:latest

# Deploy to Cloud Run
gcloud run deploy housing-price-predictor \
    --image us-central1-docker.pkg.dev/logical-vim-472815-h7/housing-price-predictor/housing-price-predictor:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --min-instances 0 \
    --port 8080
```

## 📱 Accessing Your Application

After deployment, you'll get:

- **🌐 Web Interface**: `https://housing-price-predictor-[hash]-uc.a.run.app`
- **🔧 API Endpoint**: `https://housing-price-predictor-[hash]-uc.a.run.app/predict`
- **❤️ Health Check**: `https://housing-price-predictor-[hash]-uc.a.run.app/health`

## 🔄 Updating Your Application

### Automatic Updates (via Git)
If you've set up Cloud Build triggers:
```bash
# Push changes to your main branch
git add .
git commit -m "Update housing predictor"
git push origin main
# Cloud Build will automatically deploy
```

### Manual Updates
```bash
# Make changes to your code
# Then run the deployment script again
./deploy.sh
```

## 📊 Monitoring and Logs

### View Logs
```bash
# View recent logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=housing-price-predictor" --limit 50

# Follow logs in real-time
gcloud logs tail "resource.type=cloud_run_revision AND resource.labels.service_name=housing-price-predictor"
```

### Monitor Performance
- Visit: https://console.cloud.google.com/run?project=logical-vim-472815-h7
- View metrics, logs, and performance data

## 💰 Cost Optimization

Your deployment is optimized for cost:

- **Serverless**: Pay only for actual usage
- **Auto-scaling**: Scales to zero when not used
- **Efficient**: 1GB RAM, 1 CPU sufficient for your model
- **Regional**: us-central1 has good pricing

Expected monthly cost: **$5-20** (depending on usage)

## 🔒 Security Features

- ✅ HTTPS enabled by default
- ✅ Non-root Docker user
- ✅ Health checks configured
- ✅ Resource limits set
- ✅ Public access (as requested)

## 🛠️ Troubleshooting

### Common Issues

1. **"Permission denied" errors**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

2. **"API not enabled" errors**
   ```bash
   gcloud services enable [API_NAME]
   ```

3. **Docker build failures**
   ```bash
   # Check Docker is running
   docker --version
   docker run hello-world
   ```

4. **Service not accessible**
   ```bash
   # Check service status
   gcloud run services describe housing-price-predictor --region=us-central1
   ```

### Getting Help

- **GCP Console**: https://console.cloud.google.com/run?project=logical-vim-472815-h7
- **Cloud Run Docs**: https://cloud.google.com/run/docs
- **Terraform Docs**: https://registry.terraform.io/providers/hashicorp/google/latest/docs

## 📈 Scaling

Your service automatically scales based on traffic:

- **Min instances**: 0 (saves money when idle)
- **Max instances**: 10 (handles traffic spikes)
- **Auto-scaling**: Based on CPU and memory usage

To adjust scaling, modify `terraform.tfvars` and redeploy.

## 🎯 Next Steps

1. **Custom Domain**: Add your own domain name
2. **API Authentication**: Add API key authentication
3. **Monitoring**: Set up alerts and dashboards
4. **CI/CD**: Automate deployments with GitHub Actions
5. **Caching**: Add Redis for improved performance

---

**🎉 Congratulations!** Your Housing Price Predictor is now running on Google Cloud Platform!
