#!/bin/bash

# Deployment script for Housing Price Predictor on Google Cloud Platform
# This script automates the entire deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="logical-vim-472815-h7"
REGION="us-central1"
SERVICE_NAME="housing-price-predictor"
IMAGE_NAME="housing-price-predictor"

echo -e "${BLUE}🏠 Housing Price Predictor - GCP Deployment${NC}"
echo -e "${BLUE}============================================${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed. Please install it first:${NC}"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install it first:${NC}"
    echo "https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}❌ Terraform is not installed. Please install it first:${NC}"
    echo "https://learn.hashicorp.com/tutorials/terraform/install-cli"
    exit 1
fi

echo -e "${YELLOW}📋 Pre-deployment checks passed!${NC}"

# Set the project
echo -e "${BLUE}🔧 Setting GCP project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${BLUE}🔌 Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Configure Docker authentication
echo -e "${BLUE}🐳 Configuring Docker authentication...${NC}"
gcloud auth configure-docker $REGION-docker.pkg.dev

# Deploy infrastructure with Terraform
echo -e "${BLUE}🏗️  Deploying infrastructure with Terraform...${NC}"
cd terraform
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -var-file="terraform.tfvars" -auto-approve

# Get the repository name from Terraform output
REPO_NAME=$(terraform output -raw artifact_registry_url | cut -d'/' -f2)

# Go back to project root
cd ..

# Build and push Docker image
echo -e "${BLUE}🐳 Building and pushing Docker image...${NC}"
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME/$IMAGE_NAME:latest .

echo -e "${BLUE}📤 Pushing Docker image...${NC}"
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME/$IMAGE_NAME:latest

# Deploy to Cloud Run
echo -e "${BLUE}🚀 Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME/$IMAGE_NAME:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --min-instances 0 \
    --port 8080

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🏠 Housing Price Predictor is now live!${NC}"
echo -e "${GREEN}📱 Web Interface: $SERVICE_URL${NC}"
echo -e "${GREEN}🔧 API Endpoint: $SERVICE_URL/predict${NC}"
echo -e "${GREEN}❤️  Health Check: $SERVICE_URL/health${NC}"
echo ""
echo -e "${BLUE}📊 Monitor your service:${NC}"
echo -e "${BLUE}https://console.cloud.google.com/run?project=$PROJECT_ID${NC}"
echo ""
echo -e "${YELLOW}🔄 To update the service:${NC}"
echo -e "${YELLOW}1. Make changes to your code${NC}"
echo -e "${YELLOW}2. Run: ./deploy.sh${NC}"
