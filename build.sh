#!/bin/bash

# 1. Update the Dockerfile based on devbox.json
echo "🔄 Generating Dockerfile from Devbox..."
devbox generate dockerfile

# 2. Build the Docker image
# Using --platform linux/amd64 ensures compatibility with most cloud providers
echo "🐳 Building Docker image..."
docker build -t my-python-app:latest .

echo "✅ Build complete. You can now push 'my-python-app:latest' to your registry."