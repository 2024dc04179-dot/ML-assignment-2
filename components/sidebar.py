"""Sidebar components"""

import streamlit as st
import os
from utils.data_processing import generate_test_data_if_needed
from utils.metrics import get_model_results


def render_sidebar(page):
    """Render sidebar content based on selected page"""
    uploaded_file = None
    model_file = None
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
                help="Choose a classification model. XGBoost typically performs best (92%+ accuracy)"
            )
            
            all_results = get_model_results()
            if selected_display in all_results:
                model_info = all_results[selected_display]
                st.info(f"📊 Expected Accuracy: {model_info['Accuracy']*100:.2f}% | AUC: {model_info['AUC']:.4f}")
            model_key = model_mapping[selected_display]
            
            st.markdown("---")
            st.markdown("### 📤 Upload Model File")
            
            model_file = st.file_uploader(
                "Upload model file (.py or .ipynb)",
                type=['py', 'ipynb'],
                help="Upload your model training code (.py) or notebook (.ipynb) for evaluation",
                key="model_file_uploader"
            )
            
            if model_file:
                size_kb = len(model_file.getvalue()) / 1024
                file_type = "Python script" if model_file.name.endswith('.py') else "Jupyter notebook"
                st.success(f"✓ {model_file.name} ({size_kb:.1f} KB) - {file_type}")
            
            st.markdown("---")
            st.markdown("### 📂 Upload Test Data")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload test data with 'target' or 'GradeClass' column. Must have 14 features matching training data."
            )
            
            if uploaded_file:
                size_kb = len(uploaded_file.getvalue()) / 1024
                st.success(f"✓ {uploaded_file.name} ({size_kb:.1f} KB)")
            
            st.markdown("---")
            st.markdown("### 📥 Download Test Data")
            
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    
    return page, uploaded_file, model_file, model_key, selected_display


