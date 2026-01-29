# Deployment Guide for Streamlit Cloud

## Prerequisites

1. **GitHub Account** - Create one at https://github.com if you don't have it
2. **Hugging Face Account** - Create free account at https://huggingface.co/join

## Step 1: Get Your Hugging Face API Key

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it something like "streamlit-rankings-dashboard"
4. Select "Read" access (free)
5. Copy the token (starts with `hf_...`)

## Step 2: Prepare Your Repository

### Option A: Using Git (Recommended)

```bash
# Initialize git repository
git init

# Create .gitignore file
echo ".streamlit/secrets.toml" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

# Add all files
git add .

# Commit
git commit -m "Initial commit - University Rankings Dashboard"

# Create GitHub repository (go to github.com/new)
# Then push your code
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Option B: Upload to GitHub Manually

1. Go to https://github.com/new
2. Create a new repository
3. Upload these files:
   - `main.py`
   - `chatbot_hf.py`
   - `requirements.txt`
   - `peer.csv`
   - `TIMES.xlsx`
   - `QS.xlsx`
   - `USN.xlsx`
   - `Washington.xlsx`
   - `CLAUDE.md`
   - **DO NOT upload `.streamlit/secrets.toml`**

## Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set:
   - **Main file path**: `main.py`
   - **Python version**: 3.11 or higher

## Step 4: Configure Secrets in Streamlit Cloud

1. In your app settings, find "Secrets" section
2. Add your API key in TOML format:

```toml
HF_API_KEY = "hf_your_actual_token_here"
```

3. Click "Save"

## Step 5: Deploy!

Click "Deploy" and wait for your app to launch (usually 2-3 minutes).

## Testing Your Deployment

Once deployed, test the chatbot:
1. Select a model:
   - **Qwen-2.5-72B**: Best quality, more accurate (recommended)
   - **Llama-3.3-70B**: Fast and reliable alternative
2. Select a dataset (TIMES, QS, USN, or Washington)
3. Ask a question like: "Which university has the best rank?"

## Troubleshooting

### "Error: Invalid API key"
- Regenerate your HF token and update it in Streamlit secrets

### "Error: Rate limit exceeded"
- Free HF API has rate limits. Wait a few minutes and try again
- Try the alternative model (if using Qwen, switch to Llama)

### "Module not found" errors
- Check that `requirements.txt` is in your repository
- Verify all dependencies are listed

### Data files not found
- Ensure all .xlsx files are committed to GitHub
- Check file names match exactly (case-sensitive)

## Updating Your App

To update after deployment:

```bash
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will automatically redeploy.

## Free Tier Limits

- **Streamlit Cloud**: Unlimited public apps, 1GB RAM per app
- **Hugging Face Inference API**: Free tier with rate limits
  - ~1000 requests/day for most models
  - Consider upgrading for production use

## Security Checklist

✅ `.streamlit/secrets.toml` is in `.gitignore`
✅ No API keys in code
✅ Secrets configured in Streamlit Cloud dashboard
✅ Repository can be public (no sensitive data)

## Support

- Streamlit Docs: https://docs.streamlit.io/
- HuggingFace Docs: https://huggingface.co/docs/api-inference/
- Community Forum: https://discuss.streamlit.io/
