# OpenAI API Setup Guide for Q&A Feature

## Quick Setup (3 Steps)

### Step 1: Get Your OpenAI API Key

1. Go to: https://platform.openai.com/api-keys
2. Sign up or log in to your OpenAI account
3. Click **"Create new secret key"**
4. Give it a name (e.g., "Stock Trading App")
5. **Copy the API key immediately** (you won't see it again!)
   - It looks like: `sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Step 2: Set the API Key as Environment Variable

**For Windows PowerShell (Recommended):**
```powershell
$env:OPENAI_API_KEY="sk-proj-your-actual-api-key-here"
```

**For Windows Command Prompt (CMD):**
```cmd
set OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

**Important:** Replace `sk-proj-your-actual-api-key-here` with your actual API key!

### Step 3: Restart the Streamlit App

1. **Stop the current app:**
   - Go to your PowerShell/CMD window
   - Press `Ctrl + C`
   - Type `Y` if asked to confirm

2. **Start the app again:**
   ```powershell
   python -m streamlit run app.py
   ```

3. **Verify it's working:**
   - Go to the **"💬 AI Assistant"** tab
   - You should see: **"✅ AI Assistant Ready"** (green message)
   - If you see a warning, the API key isn't set correctly

---

## Testing the Q&A Feature

1. **Make sure you've run the analysis first:**
   - Upload CSV file
   - Click "🚀 Run Complete Analysis"
   - Wait for "✅ Analysis complete!"

2. **Go to AI Assistant tab:**
   - Click on "💬 AI Assistant" tab

3. **Ask a question:**
   - Type a question like: "What does the RSI indicate about the current trend?"
   - Click "Ask AI" button
   - Wait for the AI response

---

## Troubleshooting

### Problem: "OpenAI API Key Not Set" warning

**Solution:**
- Make sure you set the environment variable in the SAME terminal window where you run Streamlit
- The environment variable only lasts for that terminal session
- Restart the app after setting the key

### Problem: "Error querying LLM" or API errors

**Possible causes:**
1. **Invalid API key:**
   - Make sure you copied the entire key correctly
   - Check for extra spaces

2. **No API credits:**
   - Check your OpenAI account balance
   - Go to: https://platform.openai.com/account/billing

3. **Network issues:**
   - Check your internet connection
   - Try again after a few seconds

### Problem: API key works but disappears after closing terminal

**Solution:**
The environment variable only lasts for that terminal session. To make it permanent:

**Option 1: Set it each time (Simple)**
- Just run the `$env:OPENAI_API_KEY="..."` command each time before starting the app

**Option 2: Create a startup script (Permanent)**
1. Create a file called `start_app.bat` in your project folder
2. Add this content:
   ```batch
   @echo off
   set OPENAI_API_KEY=sk-proj-your-actual-api-key-here
   python -m streamlit run app.py
   ```
3. Replace `sk-proj-your-actual-api-key-here` with your actual key
4. Double-click `start_app.bat` to run the app

---

## Cost Information

- **Free tier:** OpenAI provides $5 free credits for new accounts
- **Cost:** Approximately $0.002 per question (very cheap!)
- **Monitor usage:** https://platform.openai.com/usage

---

## Security Note

⚠️ **Never share your API key or commit it to GitHub!**

- The app reads the key from environment variables only
- Never hardcode your API key in the code
- If you accidentally share it, revoke it immediately at: https://platform.openai.com/api-keys

---

## Quick Reference

```powershell
# Set API key (PowerShell)
$env:OPENAI_API_KEY="your-key-here"

# Set API key (CMD)
set OPENAI_API_KEY=your-key-here

# Run app
python -m streamlit run app.py
```

---

## Summary

1. Get API key from https://platform.openai.com/api-keys
2. Set it: `$env:OPENAI_API_KEY="your-key"`
3. Restart app: `python -m streamlit run app.py`
4. Go to "💬 AI Assistant" tab
5. Ask questions!

That's it! 🎉

