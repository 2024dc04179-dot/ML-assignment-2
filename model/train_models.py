"""
ML Assignment 2 - Model Training Script
Trains all 6 classification models and saves them along with evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
import joblib
import os

# Create model directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
saved_models_dir = os.path.join(project_root, 'saved_models')
os.makedirs(saved_models_dir, exist_ok=True)

def load_and_prepare_data(file_path='../data/Student_Performance_data.csv'):
    """
    Load and prepare the dataset for training
    """
    # Try different possible paths
    possible_paths = [
        file_path,
        'data/Student_Performance_data.csv',
        '../data/Student_Performance_data.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'Student_Performance_data.csv')
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError(
            f"Dataset not found. \n"
            f"Tried paths: {possible_paths}"
        )
    
    # Separate features and target
    X = df.drop('GradeClass', axis=1)
    y = df['GradeClass']
    
    # Note: Categorical features are already encoded in the processed dataset
    # If there are still categorical features, encode them
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"Encoding remaining categorical features: {categorical_cols}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (important for Logistic Regression, KNN, and Naive Bayes)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train Logistic Regression model"""
    print("Training Logistic Regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)  # Keep full 2D array for multiclass
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    model_path = os.path.join(saved_models_dir, 'logistic_regression.pkl')
    joblib.dump(model, model_path)
    
    return model, metrics

def train_decision_tree(X_train, X_test, y_train, y_test):
    """Train Decision Tree Classifier"""
    print("Training Decision Tree...")
    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Keep full 2D array for multiclass
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    model_path = os.path.join(saved_models_dir, 'decision_tree.pkl')
    joblib.dump(model, model_path)
    
    return model, metrics

def train_knn(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train K-Nearest Neighbor Classifier"""
    print("Training KNN...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)  # Keep full 2D array for multiclass
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    model_path = os.path.join(saved_models_dir, 'knn.pkl')
    joblib.dump(model, model_path)
    
    return model, metrics

def train_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train Naive Bayes Classifier"""
    print("Training Naive Bayes...")
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)  # Keep full 2D array for multiclass
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    model_path = os.path.join(saved_models_dir, 'naive_bayes.pkl')
    joblib.dump(model, model_path)
    
    return model, metrics

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest Classifier"""
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Keep full 2D array for multiclass
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    model_path = os.path.join(saved_models_dir, 'random_forest.pkl')
    joblib.dump(model, model_path)
    
    return model, metrics

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost Classifier"""
    print("Training XGBoost...")
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Keep full 2D array for multiclass
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    model_path = os.path.join(saved_models_dir, 'xgboost.pkl')
    joblib.dump(model, model_path)
    
    return model, metrics

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics for multiclass classification"""
    # Ensure y_pred_proba is 2D (n_samples, n_classes) for multiclass
    if y_pred_proba.ndim == 1:
        # If 1D, it means binary - convert to 2D
        y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
    
    # For multiclass classification
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def main():
    """Main function to train all models"""
    print("Loading and preparing data...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_prepare_data()
    
    # Save scaler for later use
    scaler_path = os.path.join(saved_models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Train all models
    results = {}
    
    lr_model, lr_metrics = train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
    results['Logistic Regression'] = lr_metrics
    
    dt_model, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test)
    results['Decision Tree'] = dt_metrics
    
    knn_model, knn_metrics = train_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    results['kNN'] = knn_metrics
    
    nb_model, nb_metrics = train_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test)
    results['Naive Bayes'] = nb_metrics
    
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    results['Random Forest'] = rf_metrics
    
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    results['XGBoost'] = xgb_metrics
    
    # Save results to Excel
    results_df = pd.DataFrame(results).T
    results_df.columns = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    results_path = os.path.join(project_root, 'ML_Assignment_2_Results.xlsx')
    results_df.to_excel(results_path, index=True)
    
    print("\n" + "="*50)
    print(f"Training Complete! Results saved to {results_path}")
    print("="*50)
    print("\nResults Summary:")
    print(results_df.round(4))
    
    return results

if __name__ == "__main__":
    main()

