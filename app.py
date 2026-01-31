"""
ML Assignment 2 - Student Performance Classification
Streamlit Web Application for Model Evaluation
Author: Abhishek Anand (2024DC04179)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Keep sidebar always visible
st.markdown("""
<style>
    /* Sidebar always visible */
    section[data-testid="stSidebar"] {
        min-width: 280px !important;
        width: 280px !important;
        transform: none !important;
        position: relative !important;
        visibility: visible !important;
    }
    
    /* Hide sidebar collapse button */
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* Sidebar content visibility */
    section[data-testid="stSidebar"] > div {
        width: 100% !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1200px;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.35);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
    }
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0 0 0.5rem 0;
        position: relative;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.05rem;
        font-weight: 500;
        position: relative;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #181825 100%) !important;
    }
    
    
    /* ALL sidebar text white */
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] div {
        color: #cdd6f4 !important;
    }
    
    section[data-testid="stSidebar"] h2 {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #6366f1;
        margin-bottom: 1rem !important;
    }
    
    section[data-testid="stSidebar"] h3 {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #a6adc8 !important;
        margin: 1.5rem 0 0.75rem 0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: #313244 !important;
        margin: 1.25rem 0 !important;
    }
    
    /* Sidebar Radio Buttons */
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        color: #cdd6f4 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        background: #313244 !important;
        padding: 0.85rem 1rem !important;
        border-radius: 10px !important;
        margin: 5px 0 !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #45475a !important;
        border-color: #6366f1 !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border-color: transparent !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Sidebar Select Box */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        color: #cdd6f4 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background: #313244 !important;
        border: 2px solid #45475a !important;
        border-radius: 10px !important;
        transition: border-color 0.2s !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:hover {
        border-color: #6366f1 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div > div {
        color: #ffffff !important;
    }
    
    /* Sidebar File Uploader */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #313244 !important;
        border: 2px dashed #6366f1 !important;
        border-radius: 14px !important;
        padding: 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        background: #3b3d54 !important;
        border-color: #8b5cf6 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        color: #bac2de !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.7rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Sidebar Download Button - Compact */
    section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.35rem 0.75rem !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        box-shadow: 0 2px 4px rgba(99,102,241,0.2) !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%) !important;
    }
    
    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.05);
        border-left: 5px solid #6366f1;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .info-card h3 {
        color: #1e293b !important;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-card p {
        color: #475569 !important;
        line-height: 1.7;
        margin: 0;
        font-size: 0.95rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1rem 0;
        border-left: 5px solid #10b981;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    .success-card h4 {
        color: #065f46 !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .success-card p {
        color: #047857 !important;
        font-size: 0.95rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
        border: 1px solid rgba(99, 102, 241, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .metric-icon { font-size: 1.75rem; margin-bottom: 0.5rem; }
    .metric-value { font-size: 1.8rem; font-weight: 800; font-family: 'Fira Code', monospace; background: linear-gradient(135deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .metric-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-top: 0.3rem; }
    
    /* Section headers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b !important;
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid transparent;
        background: linear-gradient(90deg, #6366f1, #8b5cf6) padding-box, linear-gradient(90deg, #6366f1, #8b5cf6) border-box;
        border-image: linear-gradient(90deg, #6366f1, #8b5cf6) 1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Model badges */
    .model-badge {
        display: inline-block;
        padding: 0.6rem 1.25rem;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: #ffffff !important;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
        transition: all 0.25s ease;
    }
    
    .model-badge:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        padding: 8px;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
        color: #6366f1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
    }
    
    /* Streamlit metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    [data-testid="stMetric"] label { 
        color: #64748b !important; 
        font-weight: 600; 
        text-transform: uppercase; 
        font-size: 0.7rem; 
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] { 
        color: #1e293b !important; 
        font-family: 'Fira Code', monospace; 
        font-weight: 700; 
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] { 
        background: #ffffff; 
        border-radius: 14px; 
        overflow: hidden; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    /* Expanders */
    [data-testid="stExpander"] { 
        background: #ffffff; 
        border: 1px solid #e2e8f0; 
        border-radius: 14px;
        overflow: hidden;
        transition: all 0.2s ease;
    }
    
    [data-testid="stExpander"]:hover {
        border-color: #6366f1;
    }
    
    [data-testid="stExpander"] summary { 
        color: #1e293b; 
        font-weight: 600; 
    }
    
    /* Markdown */
    .main .stMarkdown h1 { color: #0f172a !important; font-weight: 800; font-size: 1.75rem; }
    .main .stMarkdown h2 { color: #1e293b !important; font-weight: 700; font-size: 1.4rem; }
    .main .stMarkdown h3 { color: #334155 !important; font-weight: 700; font-size: 1.15rem; }
    .main .stMarkdown p, .main .stMarkdown li { color: #475569 !important; line-height: 1.7; font-size: 0.95rem; }
    .main .stMarkdown strong { color: #1e293b !important; font-weight: 600; }
    .main .stMarkdown code { 
        background: linear-gradient(135deg, #f1f5f9, #e2e8f0); 
        color: #6366f1; 
        padding: 0.2rem 0.5rem; 
        border-radius: 6px; 
        font-family: 'Fira Code', monospace; 
        font-size: 0.85em;
        border: 1px solid #e2e8f0;
    }
    
    /* Alerts */
    .stSuccess { 
        background: linear-gradient(135deg, #ecfdf5, #d1fae5) !important; 
        border: 1px solid #a7f3d0 !important; 
        border-radius: 12px;
        border-left: 4px solid #10b981 !important;
    }
    .stError { 
        background: linear-gradient(135deg, #fef2f2, #fee2e2) !important; 
        border: 1px solid #fecaca !important; 
        border-radius: 12px;
        border-left: 4px solid #ef4444 !important;
    }
    .stInfo { 
        background: linear-gradient(135deg, #eff6ff, #dbeafe) !important; 
        border: 1px solid #bfdbfe !important; 
        border-radius: 12px;
        border-left: 4px solid #3b82f6 !important;
    }
    
    /* Hide branding */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #94a3b8, #64748b); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #6366f1, #8b5cf6); }
    
    /* Animations */
    .animate-fade { animation: fadeIn 0.6s ease-out; }
    @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(15px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    
    /* Pulse glow effect */
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 5px rgba(99, 102, 241, 0.3); }
        50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.5); }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name):
    """Load trained model"""
    paths = [
        f'saved_models/{model_name}.pkl',
        os.path.join(os.path.dirname(__file__), 'saved_models', f'{model_name}.pkl')
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
        os.path.join(os.path.dirname(__file__), 'saved_models', 'scaler.pkl')
    ]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None


def get_test_data_bytes():
    """Get test data file bytes for download"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    test_data_path = os.path.join(data_dir, 'test_data.csv')
    
    if not os.path.exists(test_data_path):
        return None
    
    try:
        with open(test_data_path, 'rb') as f:
            data = f.read()
        return data
    except Exception as e:
        return None


def generate_test_data_if_needed():
    """Generate test data if missing"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    test_data_path = os.path.join(data_dir, 'test_data.csv')
    
    if os.path.exists(test_data_path):
        return True
    
    try:
        os.makedirs(data_dir, exist_ok=True)
        
        train_paths = [
            os.path.join(data_dir, 'Student_performance_data.csv'),
            os.path.join(script_dir, 'data', 'Student_performance_data.csv'),
            'data/Student_performance_data.csv'
        ]
        
        train_df = None
        for path in train_paths:
            if os.path.exists(path):
                train_df = pd.read_csv(path)
                break
        
        if train_df is None:
            return False
        
        train_features = train_df.drop('GradeClass', axis=1)
        n_samples = 500
        
        np.random.seed(42)
        test_data = {}
        
        start_id = train_df['StudentID'].max() + 1
        test_data['StudentID'] = range(start_id, start_id + n_samples)
        test_data['Age'] = np.random.choice(train_df['Age'].values, size=n_samples, replace=True)
        
        for col in ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 
                   'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']:
            probs = train_df[col].value_counts(normalize=True).sort_index()
            test_data[col] = np.random.choice(probs.index.values, size=n_samples, p=probs.values, replace=True)
        
        mean_st = train_df['StudyTimeWeekly'].mean()
        std_st = train_df['StudyTimeWeekly'].std()
        test_data['StudyTimeWeekly'] = np.maximum(0, np.random.normal(mean_st, std_st, n_samples))
        test_data['Absences'] = np.random.choice(train_df['Absences'].values, size=n_samples, replace=True)
        
        mean_gpa = train_df['GPA'].mean()
        std_gpa = train_df['GPA'].std()
        test_data['GPA'] = np.clip(np.random.normal(mean_gpa, std_gpa, n_samples), 0, 4.0)
        
        test_df = pd.DataFrame(test_data)
        test_df = test_df[train_features.columns]
        
        test_df.to_csv(test_data_path, index=False)
        return True
        
    except Exception as e:
        st.error(f"Error generating test data: {str(e)}")
        return False


def get_model_results():
    """Get pre-computed model results"""
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


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    labels = [f'Grade {int(c)}' for c in classes]
    
    cmap = sns.light_palette("#6366f1", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold', 'color': '#1e293b'})
    
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold', color='#1e293b', labelpad=10)
    ax.set_ylabel('Actual', fontsize=13, fontweight='bold', color='#1e293b', labelpad=10)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15, color='#1e293b')
    
    ax.tick_params(colors='#475569', labelsize=11)
    plt.setp(ax.get_xticklabels(), color='#475569', fontweight='500')
    plt.setp(ax.get_yticklabels(), color='#475569', fontweight='500')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='#475569')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#475569')
    cbar.set_label('Count', color='#475569', fontweight='600')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results_dict):
    """Compare model metrics"""
    df = pd.DataFrame(results_dict).T
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor('#ffffff')
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
        ax.set_facecolor('#f8fafc')
        
        values = df[metric].values
        models = df.index.tolist()
        
        bars = ax.barh(models, values, color=colors[idx], edgecolor='white', linewidth=2,
                      alpha=0.85, height=0.6)
        
        ax.set_xlabel(metric, fontsize=11, fontweight='bold', color='#1e293b')
        ax.set_xlim(0, 1.1)
        ax.axvline(x=values.max(), color='#22c55e', linestyle='--', alpha=0.8, linewidth=2)
        
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=10, color='#1e293b',
                   fontweight='600')
        
        ax.tick_params(colors='#475569', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.spines['left'].set_color('#e2e8f0')
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#cbd5e1')
        plt.setp(ax.get_yticklabels(), color='#475569', fontsize=10, fontweight='500')
        plt.setp(ax.get_xticklabels(), color='#64748b')
    
    plt.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', 
                 y=0.98, color='#1e293b')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    st.markdown("""
    <div class="main-header animate-fade">
        <h1 class="main-title">📚 Student Performance Classification</h1>
        <p class="subtitle">Machine Learning Assignment 2 | Abhishek Anand (2024DC04179)</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = None
    model_key = None
    selected_display = None
    
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        st.markdown("---")
        
        page = st.radio(
            "Select Page",
            ["🔬 Model Evaluation", "📊 Compare Models", "📖 Documentation"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if page == "🔬 Model Evaluation":
            st.markdown("### 🤖 Select Model")
            
            model_mapping = {
                "Logistic Regression": "logistic_regression",
                "Decision Tree": "decision_tree", 
                "K-Nearest Neighbors": "knn",
                "Naive Bayes (Gaussian)": "naive_bayes",
                "Random Forest": "random_forest",
                "XGBoost": "xgboost"
            }
            
            selected_display = st.selectbox(
                "Model",
                list(model_mapping.keys()),
                help="Choose a classification model"
            )
            model_key = model_mapping[selected_display]
            
            st.markdown("---")
            st.markdown("### 📂 Upload Data")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload test data with 'target' or 'GradeClass' column"
            )
            
            if uploaded_file:
                size_kb = len(uploaded_file.getvalue()) / 1024
                st.success(f"✓ {uploaded_file.name} ({size_kb:.1f} KB)")
            
            st.markdown("---")
            st.markdown("### 📥 Download Test Data")
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data')
            test_data_path = os.path.join(data_dir, 'test_data.csv')
            
            if not os.path.exists(test_data_path):
                with st.spinner("Generating test data..."):
                    if generate_test_data_if_needed():
                        st.success("✅ Test data generated!")
                    else:
                        st.error("❌ Failed to generate test data")
            
            if os.path.exists(test_data_path):
                try:
                    with open(test_data_path, 'rb') as f:
                        csv_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download test_data.csv",
                        data=csv_bytes,
                        file_name="test_data.csv",
                        mime="text/csv",
                        key="sidebar_download_btn"
                    )
                    st.caption("💡 500 samples, same distributions as training data")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.info("Test data not found. Will be auto-generated.")
    
    # Main content
    if page == "🔬 Model Evaluation":
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Find target column
                target_col = None
                for col in ['target', 'Target', 'GradeClass', 'gradeclass']:
                    if col in df.columns:
                        target_col = col
                        break
                
                if not target_col:
                    st.error("⚠️ Target column not found. Please ensure your CSV has 'target' or 'GradeClass' column.")
                    return
                
                st.markdown('<p class="section-header">📋 Dataset Overview</p>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Features", f"{df.shape[1] - 1}")
                with col3:
                    st.metric("Classes", f"{df[target_col].nunique()}")
                with col4:
                    st.metric("Target", target_col)
                
                with st.expander("🔍 View Data Sample"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Prepare data
                if target_col != 'target':
                    df = df.rename(columns={target_col: 'target'})
                
                X = df.drop('target', axis=1)
                y = df['target']
                
                # Load model
                model = load_model(model_key)
                scaler = load_scaler()
                
                if model is None:
                    st.error(f"❌ Model '{selected_display}' not found in saved_models/")
                    return
                
                # Apply scaling for models that need it
                scale_models = ['logistic_regression', 'knn', 'naive_bayes']
                if model_key in scale_models and scaler:
                    X_processed = scaler.transform(X)
                else:
                    X_processed = X
                
                # Get predictions
                y_pred = model.predict(X_processed)
                y_pred_proba = model.predict_proba(X_processed)
                
                # Calculate metrics
                metrics = calculate_metrics(y, y_pred, y_pred_proba)
                
                if metrics:
                    st.markdown(f'<p class="section-header">📈 {selected_display} - Results</p>', unsafe_allow_html=True)
                    
                    # Display metrics in cards
                    cols = st.columns(6)
                    metric_icons = ['🎯', '📊', '✅', '🔍', '⚖️', '📐']
                    metric_colors = ['#6366f1', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899', '#06b6d4']
                    
                    for col, (name, value), icon, color in zip(cols, metrics.items(), metric_icons, metric_colors):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-icon">{icon}</div>
                                <div class="metric-value" style="color: {color}">{value:.4f}</div>
                                <div class="metric-label">{name}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Confusion Matrix and Classification Report
                    st.markdown('<p class="section-header">🔍 Detailed Analysis</p>', unsafe_allow_html=True)
                    
                    tab1, tab2 = st.tabs(["📊 Confusion Matrix", "📋 Classification Report"])
                    
                    with tab1:
                        fig = plot_confusion_matrix(y, y_pred, f"Confusion Matrix - {selected_display}")
                        st.pyplot(fig)
                    
                    with tab2:
                        report = classification_report(y, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check your CSV file format.")
        
        else:
            # Welcome screen
            st.markdown("""
            <div class="info-card animate-fade">
                <h3>👋 Welcome to the Student Performance Classifier</h3>
                <p>This application evaluates 6 machine learning models trained on the Student Performance dataset. 
                Upload your test data to see how well each model performs.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🚀 Quick Start
                1. **Select a model** from the sidebar dropdown
                2. **Upload your test CSV** file
                3. **View results** - metrics, confusion matrix, and report
                
                ### 📁 Data Requirements
                - CSV format with headers
                - Must include target column (`GradeClass` or `target`)
                - 14 features matching training data
                """)
            
            with col2:
                st.markdown("### 🤖 Available Models")
                models = ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", 
                         "Naive Bayes", "Random Forest", "XGBoost"]
                for m in models:
                    st.markdown(f'<span class="model-badge">{m}</span>', unsafe_allow_html=True)
                
                st.markdown("""
                
                ### 📊 Metrics Computed
                Accuracy, AUC Score, Precision, Recall, F1 Score, MCC
                """)
    
    elif page == "📊 Compare Models":
        st.markdown('<p class="section-header">📊 Model Comparison Dashboard</p>', unsafe_allow_html=True)
        
        results = get_model_results()
        
        # Summary table
        st.markdown("### 📋 Performance Summary")
        df_results = pd.DataFrame(results).T
        df_results.index.name = 'Model'
        
        st.dataframe(
            df_results.style.highlight_max(axis=0, color='#22c55e').format("{:.4f}"),
            use_container_width=True
        )
        
        # Best model indicator
        best_model = df_results['Accuracy'].idxmax()
        best_acc = df_results['Accuracy'].max()
        
        st.markdown(f"""
        <div class="success-card">
            <h4>🏆 Best Performing Model: {best_model}</h4>
            <p style="color: #94a3b8;">Achieved highest accuracy of <strong style="color: #34d399;">{best_acc:.4f}</strong> ({best_acc*100:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 📈 Visual Comparison")
        fig = plot_metrics_comparison(results)
        st.pyplot(fig)
        
        # Rankings
        st.markdown("### 🏅 Model Rankings by Metric")
        rankings = pd.DataFrame(index=df_results.index)
        for metric in df_results.columns:
            rankings[metric] = df_results[metric].rank(ascending=False).astype(int)
        rankings['Avg Rank'] = rankings.mean(axis=1).round(2)
        rankings = rankings.sort_values('Avg Rank')
        st.dataframe(rankings, use_container_width=True)
    
    elif page == "📖 Documentation":
        st.markdown('<p class="section-header">📖 Project Documentation</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📝 Problem Statement", "📊 Dataset", "🤖 Models"])
        
        with tab1:
            st.markdown("""
            ### Problem Statement
            
            The goal of this project is to predict student academic performance (grade classification) 
            based on various demographic, behavioral, and academic factors. This is a **multi-class 
            classification problem** where students are categorized into 5 grade classes (0-4).
            
            **Objective:** Build and evaluate multiple ML classification models to identify the most 
            effective approach for predicting student grades.
            
            **Applications:**
            - 🎯 Early identification of struggling students
            - 📚 Personalized learning recommendations  
            - 💰 Resource allocation optimization
            - 📈 Academic planning support
            """)
        
        with tab2:
            st.markdown("""
            ### Dataset Description
            
            **Source:** Student Performance Dataset (Kaggle)
            
            | Property | Value |
            |----------|-------|
            | Instances | 2,392 |
            | Features | 14 |
            | Target | GradeClass (5 classes: 0-4) |
            | Missing Values | None |
            
            **Features:** Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, 
            Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, 
            Volunteering, GPA
            """)
        
        with tab3:
            st.markdown("""
            ### Models Implemented
            
            | Model | Type | Key Characteristics |
            |-------|------|---------------------|
            | Logistic Regression | Linear | Interpretable, fast, baseline |
            | Decision Tree | Tree-based | Non-linear, interpretable |
            | K-Nearest Neighbors | Instance-based | Simple, no training |
            | Naive Bayes | Probabilistic | Fast, handles small data |
            | Random Forest | Ensemble (Bagging) | Robust, feature importance |
            | XGBoost | Ensemble (Boosting) | High performance, regularized |
            """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; padding: 1rem;">
            <p>Machine Learning Assignment 2</p>
            <p>Submitted by: Abhishek Anand (2024DC04179)</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
