# Quick Start Guide

Get the Student Performance Classifier running in under 5 minutes.

---

## Prerequisites

- Python 3.8+ installed
- pip package manager
- Git (optional, for cloning)

---

## Installation Steps

### Step 1: Navigate to Project Directory

```bash
cd ML-assignment-2
```

### Step 2: Create Virtual Environment

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
REM Method 1: Try python command first (if available)
python -m venv venv
venv\Scripts\activate.bat

REM Method 2: If python doesn't work, use py launcher
py -m venv venv
venv\Scripts\activate.bat

REM Method 3: Use full path to Python (most reliable - adjust path if needed)
"C:\Program Files\Python311\python.exe" -m venv venv
venv\Scripts\activate.bat
```

**Note:** If `python` is not recognized, skip to Method 2 or 3. To find your Python path, check:
- `C:\Program Files\Python*`
- `C:\Users\%USERNAME%\AppData\Local\Programs\Python*`

**Windows (PowerShell):**
```powershell
# Method 1: Try python command first (if available)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Method 2: If python doesn't work, use py launcher
py -m venv venv
.\venv\Scripts\Activate.ps1

# Method 3: Use full path to Python (most reliable - adjust path if needed)
& "C:\Program Files\Python311\python.exe" -m venv venv
.\venv\Scripts\Activate.ps1
```

**Note:** If `python` is not recognized, skip to Method 2 or 3. To find your Python path in PowerShell:
```powershell
Get-ChildItem -Path "C:\Program Files\Python*" -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue
Get-ChildItem -Path "$env:LOCALAPPDATA\Programs\Python*" -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Test Data (Optional)

```bash
python create_test_data.py
```

This creates `data/test_data.csv` with 500 samples (same distributions as training data, without target column).

**Note:** Test data is also auto-generated when you run the Streamlit app if it doesn't exist. Both `test_data.csv` and `train_models.ipynb` can be downloaded directly from the app sidebar in the "Download Files" section.

### Step 5: Train Models (Optional)

Models are pre-trained. To retrain:

**Using Python script:**
```bash
python model/train_models.py
```

**Using Jupyter notebook:**
```bash
jupyter notebook model/train_models.ipynb
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## Quick Commands Reference

| Task | Mac/Linux | Windows |
|------|-----------|---------|
| Create venv | `python3 -m venv venv` | `py -m venv venv` (or `python -m venv venv`) |
| Activate venv | `source venv/bin/activate` | `venv\Scripts\activate.bat` (cmd) or `.\venv\Scripts\Activate.ps1` (PowerShell) |
| Deactivate venv | `deactivate` | `deactivate` |
| Install deps | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Run app | `streamlit run app.py` | `streamlit run app.py` |
| Generate test data | `python create_test_data.py` | `python create_test_data.py` |
| Run tests | `python -m pytest tests/ -v` | `python -m pytest tests/ -v` |
| Train models (script) | `python model/train_models.py` | `python model/train_models.py` |
| Train models (notebook) | `jupyter notebook model/train_models.ipynb` | `jupyter notebook model/train_models.ipynb` |

---

## Troubleshooting

### "python not found" or "The specified disk or diskette cannot be accessed"
This usually means the Windows Store Python stub is broken. Try these solutions in order:

**Windows (Command Prompt):**
```cmd
REM Option 1: Use py launcher
py -m venv venv
venv\Scripts\activate.bat

REM Option 2: Use full path to Python (adjust path if needed)
"C:\Program Files\Python311\python.exe" -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
# Option 1: Use py launcher
py -m venv venv
.\venv\Scripts\Activate.ps1

# Option 2: Use full path to Python (adjust path if needed)
& "C:\Program Files\Python311\python.exe" -m venv venv
.\venv\Scripts\Activate.ps1
```

**To find your Python installation:**
```powershell
# PowerShell: Search for Python
Get-ChildItem -Path "C:\Program Files\Python*" -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue
Get-ChildItem -Path "$env:LOCALAPPDATA\Programs\Python*" -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue
```

**Mac/Linux**: Use `python3` instead of `python`

**If nothing works**: Reinstall Python from [python.org](https://www.python.org/downloads/) and check "Add Python to PATH" during installation

### "pip not found"
```bash
python -m pip install --upgrade pip
```

### XGBoost installation issues on Mac
```bash
brew install libomp
pip install xgboost
```

### Permission denied on Windows PowerShell (for Activate.ps1)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Alternative: Use Command Prompt instead of PowerShell
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

---

## Using the App

1. **Select a model** from the sidebar dropdown (6 models available, all use sklearn)
2. **Upload a CSV file** with test data (must include `GradeClass` or `target` column)
3. **View results**: metrics, confusion matrix, classification report, and model comparison
4. **Download files**: Use the "Download Files" section in sidebar to get test data or training notebook

### Test Data

**Generate test data:**
```bash
python create_test_data.py
```

**Download files from app:**
- The Streamlit app has download buttons in the sidebar under "Download Files" section:
  - Download `test_data.csv` (500 samples, no target column)
  - Download `train_models.ipynb` (training notebook)

**Test Data Format:**
Your CSV must have these columns:
- `Age`, `Gender`, `Ethnicity`, `ParentalEducation`
- `StudyTimeWeekly`, `Absences`, `Tutoring`, `ParentalSupport`
- `Extracurricular`, `Sports`, `Music`, `Volunteering`, `GPA`
- `GradeClass` (target column)

---

## Deployment to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" → Select repo → Choose `app.py` → Deploy

---

**Author:** Abhishek Anand (2024DC04179)
