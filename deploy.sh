#!/bin/bash
# deploy.sh — automated Cloud Run deployment for MeetMind
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Requirements:
#   - Google Cloud CLI installed and authenticated (gcloud auth login)
#   - GOOGLE_API_KEY set in your environment or .env file

set -e

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ID="meetmind-488903"
SERVICE_NAME="meetmind"
REGION="us-central1"
SOURCE_DIR="./app"
SECRET_NAME="meetmind-google-api-key"

# ── Validate API key ─────────────────────────────────────────────────────────
if [ -z "$GOOGLE_API_KEY" ] && [ -f "$SOURCE_DIR/.env" ]; then
  set -a
  . "$SOURCE_DIR/.env"
  set +a
fi

if [ -z "$GOOGLE_API_KEY" ]; then
  echo "Error: GOOGLE_API_KEY is not set."
  echo "Set it in your environment or in app/.env before deploying."
  exit 1
fi

echo "Deploying MeetMind to Cloud Run..."
echo "Project : $PROJECT_ID"
echo "Region  : $REGION"
echo "Service : $SERVICE_NAME"
echo ""

# ── Set project ──────────────────────────────────────────────────────────────
gcloud config set project "$PROJECT_ID"

# ── Enable required APIs ─────────────────────────────────────────────────────
echo "Enabling Cloud Run, Cloud Build and Secret Manager APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com

# ── Store the key ────────────────────────────────────────────────────────────
# Passing the key with --set-env-vars exposes it to anyone holding
# run.services.get on the project, so keep it in Secret Manager instead.
echo "Storing API key in Secret Manager..."
if ! gcloud secrets describe "$SECRET_NAME" >/dev/null 2>&1; then
  gcloud secrets create "$SECRET_NAME" --replication-policy=automatic
fi
printf %s "$GOOGLE_API_KEY" | gcloud secrets versions add "$SECRET_NAME" --data-file=-

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
  --member "serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role roles/secretmanager.secretAccessor >/dev/null

# ── Deploy ───────────────────────────────────────────────────────────────────
echo "Building and deploying..."
gcloud run deploy "$SERVICE_NAME" \
  --source "$SOURCE_DIR" \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-secrets "GOOGLE_API_KEY=$SECRET_NAME:latest"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "Deployment complete."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format "value(status.url)")
echo "Live at: $SERVICE_URL"
