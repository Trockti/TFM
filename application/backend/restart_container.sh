#!/bin/bash

# Stop and delete the container if it exists
docker stop web_anglicismos-api && docker rm -f web_anglicismos-api

# Build docker image
docker build --build-arg HF_TOKEN=$HUGGINGFACE_TOKEN -t web_anglicismos-backend .

# Run a new container
docker run -d -p 3232:3232 --name web_anglicismos-api web_anglicismos-backend

# Show real time logs
docker logs -f web_anglicismos-api