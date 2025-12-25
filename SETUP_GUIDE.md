# Complete Setup Guide - AI Stock Trading Assistant

## Step-by-Step Instructions to Run the App

### Prerequisites
- Windows 10/11 (based on your system)
- Python 3.8 or higher installed
- Internet connection (for installing packages and LLM features)

---

## Step 1: Open Command Prompt or PowerShell

1. Press `Windows Key + R`
2. Type `cmd` or `powershell` and press Enter
3. A black/blue window will open - this is your terminal

**OR**

1. Press `Windows Key`
2. Type "Command Prompt" or "PowerShell"
3. Click on the application

---

## Step 2: Navigate to Your Project Folder

In the terminal window, type these commands one by one (press Enter after each):

```powershell
cd Desktop
```

```powershell
cd "ENDSEM 5"
```

```powershell
cd PROJECT1
```

**Alternative (One Line):**
```powershell
cd "C:\Users\PRIYANKA LIMBAD\Desktop\ENDSEM 5\PROJECT1"
```

You should now see the path in your terminal showing you're in the PROJECT1 folder.

**Verify you're in the right place:**
```powershell
dir
```

You should see files like:
- app.py
- requirements.txt
- README.md
- src (folder)

---

## Step 3: Check Python Installation

Type this command:
```powershell
python --version
```

**If you see an error:**
- Python might not be installed or not in PATH
- Try: `python3 --version` or `py --version`

**If Python is not installed:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.8 or higher
3. **IMPORTANT:** During installation, check "Add Python to PATH"
4. Install it
5. Restart your terminal and try again

---

## Step 4: Install Required Packages

**IMPORTANT:** Make sure you're in the PROJECT1 folder (Step 2)

Type this command:
```powershell
pip install -r requirements.txt
```

**What this does:**
- Installs all required Python packages
- Takes 2-5 minutes depending on your internet speed
- You'll see lots of text scrolling - this is normal

**If you get a "pip is not recognized" error:**
Try:
```powershell
python -m pip install -r requirements.txt
```

**If you get permission errors:**
Try:
```powershell
python -m pip install -r requirements.txt --user
```

**Wait for it to finish** - you'll see "Successfully installed..." messages.

---

## Step 5: Run the Streamlit Application

Once packages are installed, type:
```powershell
python -m streamlit run app.py
```

**OR if that doesn't work, try:**
```powershell
python.exe -m streamlit run app.py
```

**Note:** Use `python -m streamlit` instead of just `streamlit` to avoid PATH issues.

**What happens:**
- Streamlit will start the web server
- You'll see output like:
  ```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
  ```

**The app will automatically open in your web browser!**

If it doesn't open automatically:
1. Look at the terminal for the URL (usually `http://localhost:8501`)
2. Open your web browser (Chrome, Edge, Firefox)
3. Copy and paste the URL into the address bar
4. Press Enter

---

## Step 6: Using the Application

### First Time Setup:

1. **Upload a CSV file:**
   - Click "Upload CSV file" in the left sidebar
   - Your CSV must have these columns: `date`, `open`, `high`, `low`, `close`, `volume`
   - Column names can be uppercase/lowercase - it doesn't matter

2. **Set Entry Price:**
   - Enter a price in the "Assumed Entry Price" field (default: 100.0)

3. **Run Analysis:**
   - Click the big blue "🚀 Run Complete Analysis" button
   - Wait for processing (10-30 seconds depending on data size)

4. **View Results:**
   - Scroll down to see all 6 sections:
     - Data Preview
     - Price & Indicator Charts
     - ML Prediction Summary
     - Risk Management
     - Strategy Backtesting
     - AI Q&A Assistant

### Optional: Enable AI Q&A

If you want to use the AI Q&A feature:

1. **Get OpenAI API Key:**
   - Go to https://platform.openai.com/api-keys
   - Sign up or log in
   - Create a new API key
   - Copy the key

2. **Set Environment Variable (Windows PowerShell):**
   ```powershell
   $env:OPENAI_API_KEY="your-api-key-here"
   ```
   (Replace `your-api-key-here` with your actual key)

3. **Restart Streamlit:**
   - Press `Ctrl + C` in the terminal to stop Streamlit
   - Run `streamlit run app.py` again

**Note:** The app works perfectly fine WITHOUT the API key - you just won't be able to use the AI Q&A feature.

---

## Step 7: Stopping the Application

When you're done:
1. Go back to your terminal window
2. Press `Ctrl + C`
3. Type `Y` if asked to confirm
4. The app will stop

---

## Troubleshooting

### Problem: "streamlit: The term 'streamlit' is not recognized"
**Solution:** This is a PATH issue. Use the Python module form instead:
```powershell
python -m streamlit run app.py
```

### Problem: "ModuleNotFoundError" or "No module named 'streamlit'"
**Solution:** Packages didn't install properly. Try:
```powershell
python -m pip install streamlit pandas numpy scikit-learn ta matplotlib plotly openai
```

### Problem: "Port 8501 is already in use"
**Solution:** Another Streamlit app is running. Either:
- Close the other app
- Or use a different port: `python -m streamlit run app.py --server.port 8502`

### Problem: CSV upload fails
**Solution:** Check your CSV has these exact column names (case doesn't matter):
- date
- open
- high
- low
- close
- volume

### Problem: "Analysis failed" or errors during processing
**Solution:** 
- Make sure your CSV has at least 50-100 rows of data
- Check that all price columns have valid numbers
- Ensure dates are in a recognizable format

### Problem: Charts not showing
**Solution:**
- Wait for analysis to complete
- Check browser console for errors (F12)
- Try refreshing the page

### Problem: App is slow
**Solution:**
- This is normal for large datasets
- ML training takes time (10-30 seconds)
- Be patient during "Run Complete Analysis"

---

## Quick Reference Commands

```powershell
# Navigate to project
cd "C:\Users\PRIYANKA LIMBAD\Desktop\ENDSEM 5\PROJECT1"

# Install packages
pip install -r requirements.txt

# Run app
python -m streamlit run app.py

# Stop app
Ctrl + C
```

---

## Sample CSV Format

If you need a sample CSV to test, create a file called `sample_data.csv` with:

```csv
date,open,high,low,close,volume
2024-01-01,100.0,105.0,99.0,103.0,1000000
2024-01-02,103.0,107.0,102.0,106.0,1200000
2024-01-03,106.0,108.0,104.0,105.0,1100000
```

(You'll need at least 50+ rows for the app to work properly)

---

## Need More Help?

1. Check the README.md file for detailed documentation
2. Make sure all files are in the correct locations
3. Verify Python and pip are working correctly
4. Check that you're in the right directory

---

## Summary Checklist

- [ ] Opened Command Prompt/PowerShell
- [ ] Navigated to PROJECT1 folder
- [ ] Verified Python is installed
- [ ] Installed packages with `pip install -r requirements.txt`
- [ ] Ran `streamlit run app.py`
- [ ] Opened browser to http://localhost:8501
- [ ] Uploaded CSV file
- [ ] Ran analysis
- [ ] Viewed results

**You're all set! 🎉**

