# Student Performance Classification

Multi-class classification system to predict student academic grades using machine learning.

**Name:** Abhishek Anand  
**ID:** 2024DC04179  
**Assignment:** Machine Learning - Assignment 2

---

## a. Problem Statement

This project predicts student academic performance by classifying students into grade categories (0-4) using their demographic info, study habits, and extracurricular activities.

The main goals:
- Predict one of 5 grade classes from 14 input features
- Compare 6 different ML algorithms to see which works best
- Build a web interface to evaluate models and make predictions

Could be useful for:
- Identifying students who might need extra help
- Understanding what factors affect student performance
- Helping schools allocate resources better

---

## b. Dataset Description

**Dataset:** Student Performance Data  
**Source:** [Kaggle - Student Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)

| Property | Value |
|----------|-------|
| Total Instances | 2,392 |
| Features | 14 |
| Target Variable | GradeClass (5 classes) |
| Missing Values | None |
| Train/Test Split | 80/20 (stratified) |

### Feature Details

| # | Feature | Description | Type | Range |
|---|---------|-------------|------|-------|
| 1 | Age | Student age | Numeric | 15-18 |
| 2 | Gender | Male=1, Female=0 | Binary | 0-1 |
| 3 | Ethnicity | Ethnic group code | Categorical | 0-3 |
| 4 | ParentalEducation | Highest parent education | Ordinal | 0-4 |
| 5 | StudyTimeWeekly | Hours of weekly study | Numeric | 0-20 |
| 6 | Absences | Number of absences | Numeric | 0-30 |
| 7 | Tutoring | Receives tutoring | Binary | 0-1 |
| 8 | ParentalSupport | Level of parental support | Ordinal | 0-4 |
| 9 | Extracurricular | Participates in activities | Binary | 0-1 |
| 10 | Sports | Plays sports | Binary | 0-1 |
| 11 | Music | Plays musical instrument | Binary | 0-1 |
| 12 | Volunteering | Does volunteer work | Binary | 0-1 |
| 13 | GPA | Grade Point Average | Numeric | 0-4 |
| 14 | **GradeClass** | **Target: Grade category** | **Categorical** | **0-4** |

### Target Class Distribution

| Grade Class | Description | Approximate % |
|-------------|-------------|---------------|
| 0 | Excellent (A) | ~10% |
| 1 | Good (B) | ~15% |
| 2 | Average (C) | ~20% |
| 3 | Below Average (D) | ~20% |
| 4 | Failing (F) | ~35% |

---

## c. Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.7516 | 0.8936 | 0.5965 | 0.5694 | 0.5762 | 0.6265 |
| Decision Tree | 0.9186 | 0.9170 | 0.8490 | 0.8554 | 0.8519 | 0.8794 |
| kNN | 0.6242 | 0.7750 | 0.4468 | 0.4268 | 0.4266 | 0.4363 |
| Naive Bayes | 0.7516 | 0.8987 | 0.7368 | 0.5940 | 0.5863 | 0.6380 |
| Random Forest (Ensemble) | 0.9081 | 0.9796 | 0.8810 | 0.8006 | 0.8160 | 0.8629 |
| XGBoost (Ensemble) | 0.9207 | 0.9906 | 0.8608 | 0.8546 | 0.8551 | 0.8828 |

### Model Performance Observations

| ML Model Name | Notes |
|---------------|-------|
| Logistic Regression | Got 75.16% accuracy, which is decent for a linear model. AUC is pretty high (0.8936) so it can separate classes well, but precision/recall are lower - probably struggles with similar grade classes. Good baseline model. |
| Decision Tree | Did really well at 91.86% accuracy. Handles non-linear patterns without needing scaling. Precision and recall are balanced (both around 0.85), so it's consistent across classes. Set max_depth=10 to avoid overfitting. |
| kNN | Worst performer at 62.42% accuracy. Doesn't work well with this many features and the class imbalance. Since it's distance-based, feature scaling matters a lot. Low MCC (0.4363) shows weak predictions. Might need feature selection or different k. |
| Naive Bayes | Same accuracy as Logistic Regression (75.16%) but better AUC (0.8987). The independence assumption is a problem here since GPA and study time are correlated. High precision but low recall - it's being too cautious. |
| Random Forest | Great performance at 90.81% accuracy, highest precision (0.8810). Using 100 trees helps with variance and multi-class problems. Best AUC (0.9796) among non-boosting methods. Recall is a bit lower, so some minority classes get missed. |
| XGBoost | Best model overall - 92.07% accuracy and highest AUC (0.9906). Gradient boosting learns from mistakes well. Balanced precision/recall (both ~0.86) and highest MCC (0.8828). Handles class imbalance better than others. |

---

## Project Structure

```
ML-assignment-2/
├── app.py                      # Main Streamlit web application (modular)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── QUICK_START.md              # Quick start guide
├── create_test_data.py         # Test data generation script
├── .gitignore                  # Git ignore patterns
├── ML_Assignment_2_Results.xlsx # Model evaluation metrics (generated by train_models.py)
├── ML_Assignment_2.pdf         # Assignment PDF document
│
├── data/
│   ├── Student_performance_data.csv    # Training dataset
│   ├── test_data.csv                   # Test data (500 samples, no target)
│   └── README_TEST_DATA.md            # Test data documentation
│
├── model/
│   ├── __init__.py
│   ├── train_models.py         # Model training script (.py)
│   └── train_models.ipynb      # Model training notebook (.ipynb)
│
├── saved_models/               # Trained model files
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── utils/                      # Utility modules (modular code)
│   ├── __init__.py
│   ├── model_loader.py         # Model loading functions
│   ├── data_processing.py     # Data generation utilities
│   ├── metrics.py              # Metrics calculation
│   └── visualizations.py      # Plotting functions
│
├── components/                 # UI components
│   ├── __init__.py
│   └── sidebar.py              # Sidebar component
│
└── styles/                     # Styling modules
    ├── __init__.py
    ├── css.py                  # CSS styles
    └── javascript.py           # JavaScript code
│
└── tests/
    └── test_models.py          # Unit tests
```

---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ML-assignment-2
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Test Data (Optional)

The app will auto-generate test data when you run it. If you want to generate it manually:

```bash
python create_test_data.py
```

This creates `data/test_data.csv` with 500 samples. It has the same feature distributions as the training data, but no target column.

**Note:** Both `test_data.csv` and `train_models.ipynb` are committed to GitHub and can be downloaded directly from the Streamlit app sidebar in the "Download Files" section.

### Step 5: Train Models (if needed)

The models are already trained and saved. If you want to retrain them:

**Using Python Script:**
```bash
python model/train_models.py
```

**Using Jupyter Notebook:**
```bash
jupyter notebook model/train_models.ipynb
```

Both will train all 6 models and save them to `saved_models/`. The notebook is included so you can see the training process step by step.

**Note:** All models use sklearn's built-in implementations. The training notebook (`train_models.ipynb`) can be downloaded directly from the Streamlit app sidebar. After training, model evaluation metrics are automatically saved to `ML_Assignment_2_Results.xlsx` in the project root.

### Step 6: Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Streamlit Application Features

The web app has these features:

1. **Dataset Upload** - Upload CSV test data to evaluate models
2. **Model Selection** - Pick from 6 trained models (all use sklearn's built-in implementations)
3. **Test Data Download** - Download the pre-generated test data (500 samples). The `test_data.csv` file is available on GitHub and can be downloaded directly from the Streamlit app sidebar.
4. **Model Notebook Download** - Download the training notebook (`train_models.ipynb`) directly from the app sidebar. Both the notebook and test data are available in the "Download Files" section.
5. **Metrics Display** - See Accuracy, AUC, Precision, Recall, F1, MCC
6. **Confusion Matrix** - Visual chart showing prediction accuracy
7. **Classification Report** - Detailed per-class metrics
8. **Model Comparison** - Compare all models side by side
9. **Charts and Graphs** - Interactive visualizations of model performance

### Code Structure

The code is split into modules:
- **`app.py`** - Main app file that handles the UI
- **`utils/`** - Helper functions for models, metrics, data processing, and plots
- **`components/`** - UI components like sidebar
- **`styles/`** - CSS and JavaScript for styling

---

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select repository, branch, and `app.py`
6. Click "Deploy"

**Live App:** [Link to deployed app]

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

---

## Evaluation Metrics Explained

| Metric | What it means | Range |
|--------|---------------|-------|
| **Accuracy** | How many predictions were correct | 0-1 |
| **AUC** | How well the model separates different classes | 0-1 |
| **Precision** | Of all positive predictions, how many were actually positive | 0-1 |
| **Recall** | Of all actual positives, how many did we catch | 0-1 |
| **F1 Score** | Balance between precision and recall | 0-1 |
| **MCC** | Overall correlation between predictions and actual values | -1 to 1 |

---

## Key Findings

1. XGBoost works best - 92.07% accuracy and highest MCC (0.8828)
2. Ensemble methods (Random Forest, XGBoost) beat single models by about 15-20%
3. kNN didn't work well - too many features and class imbalance hurt it
4. GPA is the most important feature for predicting grades
5. Study time and absences also matter a lot

---

## References

- Dataset: [Kaggle Student Performance](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)
- Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org)
- XGBoost Documentation: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- Streamlit Documentation: [docs.streamlit.io](https://docs.streamlit.io)

---

## License

This project is submitted as part of BITS Pilani M.Tech (AIML/DSE) coursework.

---

**Submitted by:** Abhishek Anand (2024DC04179)  
**Course:** Machine Learning  
