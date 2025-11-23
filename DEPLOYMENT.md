# Streamlit Cloud Deployment Guide

## 1. Prepare Your Repository
Ensure your project folder (`Doc&Brain`) has the following files:
- `app.py` (Main application)
- `patches.py` (Fixes for library compatibility)
- `requirements.txt` (Dependencies)
- `packages.txt` (Optional, for system packages)

## 2. Push to GitHub
1. Create a new repository on GitHub.
2. Push your code to this repository.

## 3. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Click **"New app"**.
3. Select your GitHub repository, branch (usually `main`), and file path (`app.py`).
4. Click **"Deploy!"**.

## 4. Configure Secrets (Hide API Keys)
1. Once deployed (or while deploying), go to the **"Settings"** of your app on Streamlit Cloud.
2. Click on **"Secrets"**.
3. Paste your Google API Key in the TOML format:
   ```toml
   GOOGLE_API_KEY = "your-secret-api-key-here"
   ```
4. Save. The app will restart and pick up the key automatically.

## 5. Troubleshooting
- If you see errors about `MediaResolution` or `Qdrant`, the `patches.py` file included in your project should handle them automatically.
- If the app runs out of memory, try uploading smaller PDFs or fewer files at once.
