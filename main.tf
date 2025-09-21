terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "cloud_run_api" {
  service = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "container_registry_api" {
  service = "containerregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_build_api" {
  service = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

# Create Cloud Run service
resource "google_cloud_run_v2_service" "housing_predictor" {
  name     = var.service_name
  location = var.region

  depends_on = [
    google_project_service.cloud_run_api,
    google_project_service.container_registry_api,
    google_project_service.cloud_build_api
  ]

  template {
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.image_name}:latest"
      
      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }

      env {
        name  = "PORT"
        value = "8080"
      }

      env {
        name  = "PYTHONUNBUFFERED"
        value = "1"
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 10
        timeout_seconds      = 1
        period_seconds       = 3
        failure_threshold    = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        timeout_seconds      = 1
        period_seconds       = 10
        failure_threshold    = 3
      }
    }
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
}

# Make the service publicly accessible
resource "google_cloud_run_service_iam_member" "public_access" {
  count = var.allow_unauthenticated ? 1 : 0
  
  service  = google_cloud_run_v2_service.housing_predictor.name
  location = google_cloud_run_v2_service.housing_predictor.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Create Artifact Registry repository for Docker images
resource "google_artifact_registry_repository" "housing_predictor_repo" {
  location      = var.region
  repository_id = var.service_name
  description   = "Docker repository for Housing Price Predictor"
  format        = "DOCKER"

  depends_on = [google_project_service.container_registry_api]
}

# Create Cloud Build trigger for automatic deployments
resource "google_cloudbuild_trigger" "housing_predictor_trigger" {
  name        = "${var.service_name}-trigger"
  description = "Build and deploy Housing Price Predictor"
  
  filename = "cloudbuild.yaml"
  
  github {
    owner = "brandonfosterlink"
    name  = "Cali_Housing_Prices"
    
    push {
      branch = "main"
    }
  }

  depends_on = [google_project_service.cloud_build_api]
}

# Optional: Create a custom domain mapping (uncomment if you have a domain)
# resource "google_cloud_run_domain_mapping" "housing_predictor_domain" {
#   location = var.region
#   name     = "your-domain.com"
#
#   metadata {
#     namespace = var.project_id
#   }
#
#   spec {
#     route_name = google_cloud_run_v2_service.housing_predictor.name
#   }
# }