# Terraform variables for Housing Price Predictor deployment
project_id = "logical-vim-472815-h7"
region     = "us-central1"
service_name = "housing-price-predictor"
image_name   = "housing-price-predictor"

# Performance settings
memory = "1Gi"
cpu    = "1"
max_instances = 10
min_instances = 0

# Security settings
allow_unauthenticated = true
