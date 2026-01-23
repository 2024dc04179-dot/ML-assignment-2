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
cd ML Assignment2
```

### Step 2: Create Virtual Environment

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## Quick Commands Reference

| Task | Mac/Linux | Windows |
|------|-----------|---------|
| Create venv | `python3 -m venv venv` | `python -m venv venv` |
| Activate venv | `source venv/bin/activate` | `venv\Scripts\activate` |
| Deactivate venv | `deactivate` | `deactivate` |
| Install deps | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Run app | `streamlit run app.py` | `streamlit run app.py` |
| Run tests | `python -m pytest tests/ -v` | `python -m pytest tests/ -v` |
| Train models | `python model/train_models.py` | `python model/train_models.py` |

---

## Troubleshooting

### "python not found"
- Mac: Use `python3` instead of `python`
- Windows: Ensure Python is added to PATH during installation

### "pip not found"
```bash
python -m pip install --upgrade pip
```

### XGBoost installation issues on Mac
```bash
brew install libomp
pip install xgboost
```

### Permission denied on Windows PowerShell
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Using the App

1. **Select a model** from the sidebar dropdown
2. **Upload a CSV file** with test data
3. **View results**: metrics, confusion matrix, classification report

### Test Data Format
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
