output "service_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_v2_service.housing_predictor.uri
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_v2_service.housing_predictor.name
}

output "service_location" {
  description = "Location of the Cloud Run service"
  value       = google_cloud_run_v2_service.housing_predictor.location
}

output "artifact_registry_url" {
  description = "URL of the Artifact Registry repository"
  value       = google_artifact_registry_repository.housing_predictor_repo.name
}

output "docker_push_commands" {
  description = "Commands to build and push Docker image"
  value = <<-EOT
    # Configure Docker to use gcloud as a credential helper
    gcloud auth configure-docker ${var.region}-docker.pkg.dev
    
    # Build the Docker image
    docker build -t ${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.image_name}:latest .
    
    # Push the Docker image
    docker push ${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.image_name}:latest
  EOT
}

output "access_instructions" {
  description = "Instructions for accessing the deployed service"
  value = <<-EOT
    🏠 Housing Price Predictor deployed successfully!
    
    📱 Web Interface: ${google_cloud_run_v2_service.housing_predictor.uri}
    🔧 API Endpoint: ${google_cloud_run_v2_service.housing_predictor.uri}/predict
    ❤️  Health Check: ${google_cloud_run_v2_service.housing_predictor.uri}/health
    
    🚀 To update the service:
    1. Make changes to your code
    2. Run: docker build -t ${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.image_name}:latest .
    3. Run: docker push ${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.image_name}:latest
    4. The service will automatically update
    
    📊 Monitor your service in the Google Cloud Console:
    https://console.cloud.google.com/run?project=${var.project_id}
  EOT
}
