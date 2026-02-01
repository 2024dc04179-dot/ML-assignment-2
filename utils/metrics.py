"""Metrics calculation utilities"""

import streamlit as st
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)


def get_model_results():
    """Return model evaluation results"""
    return {
        'Logistic Regression': {'Accuracy': 0.7516, 'AUC': 0.8936, 'Precision': 0.5965, 'Recall': 0.5694, 'F1': 0.5762, 'MCC': 0.6265},
        'Decision Tree': {'Accuracy': 0.9186, 'AUC': 0.9170, 'Precision': 0.8490, 'Recall': 0.8554, 'F1': 0.8519, 'MCC': 0.8794},
        'kNN': {'Accuracy': 0.6242, 'AUC': 0.7750, 'Precision': 0.4468, 'Recall': 0.4268, 'F1': 0.4266, 'MCC': 0.4363},
        'Naive Bayes': {'Accuracy': 0.7516, 'AUC': 0.8987, 'Precision': 0.7368, 'Recall': 0.5940, 'F1': 0.5863, 'MCC': 0.6380},
        'Random Forest': {'Accuracy': 0.9081, 'AUC': 0.9796, 'Precision': 0.8810, 'Recall': 0.8006, 'F1': 0.8160, 'MCC': 0.8629},
        'XGBoost': {'Accuracy': 0.9207, 'AUC': 0.9906, 'Precision': 0.8608, 'Recall': 0.8546, 'F1': 0.8551, 'MCC': 0.8828}
    }


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate classification metrics"""
    try:
        if y_pred_proba.ndim == 1:
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:
            return {
                'Accuracy': accuracy_score(y_true, y_pred),
                'AUC': roc_auc_score(y_true, y_pred_proba[:, 1]),
                'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'F1': f1_score(y_true, y_pred, average='binary', zero_division=0),
                'MCC': matthews_corrcoef(y_true, y_pred)
            }
        else:
            return {
                'Accuracy': accuracy_score(y_true, y_pred),
                'AUC': roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'),
                'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'MCC': matthews_corrcoef(y_true, y_pred)
            }
    except Exception as e:
        st.error(f"Metric calculation error: {str(e)}")
        return None


