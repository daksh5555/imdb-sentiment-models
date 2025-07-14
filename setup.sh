#!/bin/bash
# Setup Streamlit config
mkdir -p ~/.streamlit/

echo "\
[server]
port = \$PORT
enableCORS = false
headless = true
" > ~/.streamlit/config.toml

# Install and configure Git LFS
apt-get update && apt-get install -y git-lfs
git lfs install
git lfs pull
