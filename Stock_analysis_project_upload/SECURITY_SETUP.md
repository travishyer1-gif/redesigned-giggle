# Security Setup Guide for Stock Analysis Project

## 🚨 CRITICAL SECURITY ALERT - IMMEDIATE ACTION REQUIRED

**GitHub has detected your OpenAI API key in your code!** This is a serious security breach.

### ⚡ IMMEDIATE ACTIONS (Do These NOW):

1. **Revoke Your OpenAI API Key IMMEDIATELY**
   - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - Delete the exposed key (check your config.py for the actual key)
   - Generate a **NEW API key**

2. **Fix Your Configuration Structure**
   - Your current `config.py` contains real API keys
   - Multiple files import from it, exposing your keys throughout the codebase

## 🔐 New Secure Configuration Structure

### **Safe Files (Commit to Git):**
- `config_safe.py` - Contains all non-sensitive configuration
- `config_template.py` - Template for reference
- `env_template.txt` - Environment variables template
- `.gitignore` - Protects sensitive files
- All Python analysis scripts

### **Protected Files (Never Commit):**
- `config.py` - Your current file with real API keys
- `config_secrets.py` - New secrets file
- `.env` - Your actual environment variables

## 🛠️ How to Fix This Step by Step

### **Step 1: Create Your .env File**
Copy `env_template.txt` to `.env` and add your real API keys:

```bash
# Copy the template
cp env_template.txt .env

# Edit .env with your real keys
ALPHA_VANTAGE_API_KEY=sk-your-actual-key-here
OPENAI_API_KEY=sk-your-new-key-here
# ... etc
```

### **Step 2: Update Your Import Statements**
Change all files from importing `config` to importing `config_safe`:

```python
# OLD (DANGEROUS):
from config import ALPHA_VANTAGE_API_KEY, TECHNICAL_PARAMS

# NEW (SAFE):
from config_safe import TECHNICAL_PARAMS
from config_secrets import ALPHA_VANTAGE_API_KEY
```

### **Step 3: Test Your Setup**
Run your tests to ensure everything works with the new structure.

## 🛡️ What's Protected by .gitignore

The `.gitignore` file now protects:

- **Configuration files**: `config.py`, `config_secrets.py`, `.env`
- **API keys**: `*.key`, `secrets.json`, etc.
- **Cache files**: `__pycache__/`, `*.pyc`
- **Data files**: `*.csv`, `*.xlsx`, `data/`
- **Output files**: `analysis_results/`, `charts/`, `output/`

## 📋 Security Checklist

- [ ] **IMMEDIATE**: Revoke exposed OpenAI API key
- [ ] **IMMEDIATE**: Generate new OpenAI API key
- [ ] Create `.env` file with new keys
- [ ] Update all import statements to use `config_safe`
- [ ] Test that everything works
- [ ] Commit `.gitignore` and safe files
- [ ] Never commit `config.py` or `.env`

## 🚫 What NOT to Do

- ❌ **NEVER commit `config.py`** with real API keys
- ❌ **NEVER commit `.env`** files
- ❌ **NEVER hardcode API keys** in source code
- ❌ **NEVER share API keys** in public repositories

## ✅ What TO Do

- ✅ Use environment variables for sensitive data
- ✅ Import from `config_safe` for non-sensitive data
- ✅ Import from `config_secrets` for API keys
- ✅ Commit `config_safe.py` and templates
- ✅ Regularly rotate API keys

## 🔄 Setting Up on a New Machine

1. Clone the repository
2. Copy `env_template.txt` to `.env`
3. Add your API keys to `.env`
4. Install dependencies: `pip install -r requirements.txt`
5. Run your analysis scripts

## 🆘 If You Accidentally Committed API Keys

1. **IMMEDIATELY** revoke the exposed API keys
2. Generate new API keys
3. Update your `.env` file
4. Remove the file from git history:
   ```bash
   git filter-branch --force --index-filter \
   "git rm --cached --ignore-unmatch config.py" \
   --prune-empty --tag-name-filter cat -- --all
   ```
5. Force push: `git push origin --force`

## 📞 Support

If you need help with security setup or have questions about protecting your API keys, please reach out to the project maintainers.

---

**Remember: Security is everyone's responsibility. Protect your API keys!** 🔒

## 🚨 CURRENT STATUS

**YOUR PROJECT IS CURRENTLY UNSAFE** because:
- `config.py` contains real API keys
- Multiple files import from it
- GitHub has detected your OpenAI key

**Fix this immediately to prevent further exposure!**
