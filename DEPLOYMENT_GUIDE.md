# Deployment Guide - Streamlit App

## ⚠️ Important: Vercel vs Streamlit Cloud

**Vercel is NOT ideal for Streamlit apps** because:
- Vercel is designed for static sites and serverless functions
- Streamlit apps need a Python runtime that stays running
- Vercel has timeout limits that don't work well with Streamlit

## ✅ Best Option: Streamlit Cloud (FREE & EASY)

Streamlit Cloud is the **official, free hosting** for Streamlit apps. Perfect for your resume!

---

## Option 1: Streamlit Cloud (Recommended)

### Step 1: Push Your Code to GitHub

1. **Create a GitHub account** (if you don't have one):
   - Go to: https://github.com
   - Sign up for free

2. **Create a new repository:**
   - Click the "+" icon → "New repository"
   - Name it: `ai-stock-trading-assistant`
   - Make it **Public** (required for free Streamlit Cloud)
   - Click "Create repository"

3. **Upload your code to GitHub:**
   
   **Option A: Using GitHub Desktop (Easiest)**
   - Download: https://desktop.github.com
   - Install and sign in
   - Click "File" → "Add Local Repository"
   - Select your project folder: `C:\Users\PRIYANKA LIMBAD\Desktop\ENDSEM 5\PROJECT1`
   - Click "Publish repository"
   - Make sure it's **Public**

   **Option B: Using Git Command Line**
   ```bash
   cd "C:\Users\PRIYANKA LIMBAD\Desktop\ENDSEM 5\PROJECT1"
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/ai-stock-trading-assistant.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit: https://streamlit.io/cloud
   - Click "Sign up" or "Get started"
   - Sign in with your GitHub account

2. **Deploy your app:**
   - Click "New app"
   - Select your repository: `ai-stock-trading-assistant`
   - Main file path: `app.py`
   - App URL: Choose a name (e.g., `ai-stock-trading-assistant`)
   - Click "Deploy"

3. **Wait for deployment:**
   - Takes 2-5 minutes
   - You'll get a URL like: `https://ai-stock-trading-assistant.streamlit.app`

4. **Set up environment variables (for OpenAI API):**
   - In Streamlit Cloud dashboard, go to "Settings"
   - Click "Secrets"
   - Add:
     ```
     OPENAI_API_KEY = "your-actual-api-key-here"
     ```
   - Save

### Step 3: Share Your Link

Your app will be live at:
```
https://YOUR-APP-NAME.streamlit.app
```

Add this to your resume! 🎉

---

## Option 2: Alternative Platforms

### Heroku (Paid after free tier ended)

1. Create account: https://heroku.com
2. Install Heroku CLI
3. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
4. Deploy using Heroku CLI

### Railway (Free tier available)

1. Sign up: https://railway.app
2. Connect GitHub repository
3. Deploy automatically

### Render (Free tier available)

1. Sign up: https://render.com
2. Create new Web Service
3. Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run app.py --server.port=$PORT`

---

## Important Files for Deployment

Make sure these files are in your GitHub repo:

- ✅ `app.py` (main file)
- ✅ `requirements.txt` (dependencies)
- ✅ `README.md` (documentation)
- ✅ `src/` folder (all modules)
- ✅ `.gitignore` (exclude unnecessary files)

### Create `.gitignore` file:

Create a file named `.gitignore` in your project folder:

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv
*.log
.DS_Store
*.csv
*.xlsx
```

---

## Custom Domain (Optional)

Streamlit Cloud allows custom domains:
1. Go to app settings
2. Add your custom domain
3. Update DNS records

---

## Troubleshooting

### App won't deploy:
- Check `requirements.txt` has all dependencies
- Make sure `app.py` is in the root folder
- Check GitHub repo is Public

### App deploys but shows errors:
- Check Streamlit Cloud logs
- Verify all file paths are correct
- Make sure `src/` folder is included

### Environment variables not working:
- Check secrets are set correctly in Streamlit Cloud
- Restart the app after adding secrets

---

## Summary

**Easiest Path:**
1. Push code to GitHub (Public repo)
2. Sign up for Streamlit Cloud
3. Deploy from GitHub
4. Add OpenAI API key in secrets
5. Share your link!

**Your resume link will be:**
```
https://YOUR-APP-NAME.streamlit.app
```

Good luck! 🚀

