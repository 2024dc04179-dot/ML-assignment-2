"""Model loading utilities"""

import streamlit as st
import joblib
import os


@st.cache_resource
def load_model(model_name):
    """Load trained model"""
    paths = [
        f'saved_models/{model_name}.pkl',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models', f'{model_name}.pkl')
    ]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None


@st.cache_resource
def load_scaler():
    """Load feature scaler"""
    paths = [
        'saved_models/scaler.pkl',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models', 'scaler.pkl')
    ]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None


