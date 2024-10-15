#!/bin/bash

# Prompt for Githuba access token interactive
read -p "Github Access Token: " github_token

# Login with GCloud
gcloud init
gcloud auth application-default login

# Install Github CLI
sudo snap install gh

# Set Github access token
echo "Setting Github access token..."
gh auth login --with-token <<< $github_token

git config --global user.email "naterush1997@gmail.com"
git config --global user.name "Nate Rush"


# Install Google Cloud Storage Connectors
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse

# Actually attach the bucket
gcsfuse othello-gpt ./bucket