#!/bin/bash

set -e  # Exit on error

# === CONFIGURATION ===
IMAGE_NAME="ai-civ-web"
REGISTRY="aicargomation.azurecr.io"
TAG="dev4"
FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"
ACR_NAME="aicargomation"

echo "Deploying image: $FULL_IMAGE"

# === DOCKER & ACR COMMANDS ===
docker compose down -v
docker compose build

echo "Logging into Azure Container Registry..."
az acr login --name "$ACR_NAME"

echo "Tagging image..."
docker tag "$IMAGE_NAME" "$FULL_IMAGE"

echo "Pushing image..."
docker push "$FULL_IMAGE"

echo "Deployment complete!"
