# Student Performance Classification

Multi-class classification system to predict student academic grades using machine learning.

**Name:** Abhishek Anand  
**ID:** 2024DC04179  
**Assignment:** Machine Learning - Assignment 2

---

## a. Problem Statement

Predicting student academic performance is crucial for educational institutions to identify at-risk students early and provide timely interventions. This project addresses the challenge of classifying students into grade categories (0-4) based on their demographic information, study habits, and extracurricular involvement.

The classification task involves:
- Predicting one of 5 grade classes based on 14 input features
- Comparing 6 different ML algorithms to find the best approach
- Building a web interface for real-time predictions and model evaluation

This work has practical applications in:
- Early warning systems for academic advisors
- Personalized learning path recommendations
- Resource allocation for student support services

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

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieves 75.16% accuracy which is reasonable for a linear model on multi-class data. The high AUC (0.8936) indicates good class separation ability despite moderate accuracy. Lower precision and recall suggest difficulty distinguishing between adjacent grade classes. Works best as a baseline model and for interpretability when understanding feature contributions. |
| Decision Tree | Strong performer with 91.86% accuracy. Captures non-linear decision boundaries effectively without requiring feature scaling. The balanced precision (0.8490) and recall (0.8554) indicate consistent performance across all grade classes. Depth limitation (max_depth=10) prevents overfitting while maintaining predictive power. |
| kNN | Lowest performer at 62.42% accuracy. The algorithm struggles with the high-dimensional feature space and class imbalance. Being distance-based, it's sensitive to the scale and distribution of features. The low MCC (0.4363) confirms weak correlation between predictions and actual grades. May improve with feature selection or different k values. |
| Naive Bayes | Matches Logistic Regression accuracy (75.16%) with slightly better AUC (0.8987). The independence assumption doesn't fully hold for this dataset where features like GPA and StudyTime are correlated. High precision (0.7368) but lower recall (0.5940) means it's conservative in predictions, missing some correct classifications. |
| Random Forest (Ensemble) | Excellent performance at 90.81% accuracy with the highest precision (0.8810). The ensemble of 100 trees reduces variance and handles the multi-class problem well. Highest AUC among non-boosting methods (0.9796) demonstrates strong probability calibration. Slightly lower recall suggests some minority class samples are misclassified. |
| XGBoost (Ensemble) | Best overall model with 92.07% accuracy and highest AUC (0.9906). Gradient boosting effectively learns from errors of previous trees. Balanced precision (0.8608) and recall (0.8546) with highest MCC (0.8828) indicate robust performance across all metrics. The sequential error correction mechanism handles class imbalance better than other models. |

---

## Project Structure

```
ML2/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── QUICK_START.md              # Quick start guide
├── create_test_data.py         # Test data generation script
├── model_results.xlsx          # Evaluation metrics
│
├── data/
│   ├── Student_performance_data.csv    # Training dataset
│   ├── test_data.csv                   # Test data (500 samples, no target)
│   ├── validate_dataset.py            # Validation script
│   └── README_TEST_DATA.md            # Test data documentation
│
├── model/
│   ├── __init__.py
│   └── train_models.py         # Model training script
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
git clone https://github.com/yourusername/ML-Assignment-2.git
cd ML-Assignment-2
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

Test data is auto-generated when you run the app. To generate manually:

```bash
python create_test_data.py
```

This creates `data/test_data.csv` with 500 samples (same distributions as training data, without target column).

### Step 5: Train Models (if needed)

Models are pre-trained and saved. To retrain:

```bash
python model/train_models.py
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Streamlit Application Features

The web application includes:

1. **Dataset Upload (CSV)** - Upload test data for evaluation
2. **Model Selection Dropdown** - Choose from 6 trained models
3. **Test Data Download** - Download pre-generated test data (500 samples)
4. **Evaluation Metrics Display** - View Accuracy, AUC, Precision, Recall, F1, MCC
5. **Confusion Matrix** - Visual representation of predictions
6. **Classification Report** - Per-class precision, recall, and F1 scores
7. **Model Comparison Dashboard** - Compare all models side by side

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

| Metric | Description | Range |
|--------|-------------|-------|
| **Accuracy** | Proportion of correct predictions | 0-1 |
| **AUC** | Area under ROC curve; measures class separation | 0-1 |
| **Precision** | True positives / (True positives + False positives) | 0-1 |
| **Recall** | True positives / (True positives + False negatives) | 0-1 |
| **F1 Score** | Harmonic mean of precision and recall | 0-1 |
| **MCC** | Matthews Correlation Coefficient; balanced measure | -1 to 1 |

---

## Key Findings

1. **XGBoost performs best** with 92.07% accuracy and highest MCC (0.8828)
2. **Ensemble methods outperform** single classifiers by ~15-20%
3. **kNN struggles** with high dimensionality and class imbalance
4. **GPA is the strongest predictor** of grade class
5. **Study time and absences** are important secondary factors

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
