#!/bin/bash

# Prompt for Githuba access token interactive
echo "Please provide a Github access token with the following permissions:"
echo "repo, read:org, workflow"
read -p "Github Access Token: " token

# Install Github CLI
sudo snap install gh

# Set Github access token
echo "Setting Github access token..."
gh auth login --with-token <<< $token