"""
ML Assignment 2 - Student Performance Classification
Streamlit Web Application for Model Evaluation
Author: Abhishek Anand (2024DC04179)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# Import modular components
from styles.css import SIDEBAR_CSS, MAIN_CSS
from styles.javascript import HIDE_COLLAPSE_BUTTON_JS
from utils.model_loader import load_model, load_scaler
from utils.data_processing import generate_test_data_if_needed
from utils.metrics import get_model_results, calculate_metrics
from utils.visualizations import plot_confusion_matrix, plot_metrics_comparison

st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply styles from modules
st.markdown(SIDEBAR_CSS + HIDE_COLLAPSE_BUTTON_JS, unsafe_allow_html=True)
st.markdown(MAIN_CSS, unsafe_allow_html=True)




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
                help="Choose a classification model. XGBoost typically performs best (92%+ accuracy)"
            )
            
            all_results = get_model_results()
            if selected_display in all_results:
                model_info = all_results[selected_display]
                st.info(f"📊 Expected Accuracy: {model_info['Accuracy']*100:.2f}% | AUC: {model_info['AUC']:.4f}")
            model_key = model_mapping[selected_display]
            
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
            st.markdown("### 📥 Download Files")
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Download Test Data
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
                        key="sidebar_download_test_data"
                    )
                    st.caption("💡 500 samples, same distributions as training data")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.info("Test data not found. Will be auto-generated.")
            
            st.markdown("---")
            
            # Download Model Notebook
            model_dir = os.path.join(script_dir, 'model')
            notebook_path = os.path.join(model_dir, 'train_models.ipynb')
            
            if os.path.exists(notebook_path):
                try:
                    with open(notebook_path, 'rb') as f:
                        notebook_bytes = f.read()
                    
                    st.download_button(
                        label="📓 Download train_models.ipynb",
                        data=notebook_bytes,
                        file_name="train_models.ipynb",
                        mime="application/json",
                        key="sidebar_download_notebook"
                    )
                    st.caption("💡 Jupyter notebook with model training code")
                except Exception as e:
                    st.error(f"Error loading notebook: {str(e)}")
            else:
                st.info("Model notebook not found.")
    
    if page == "🔬 Model Evaluation":
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                target_col = None
                for col in ['target', 'Target', 'GradeClass', 'gradeclass']:
                    if col in df.columns:
                        target_col = col
                        break
                
                if not target_col:
                    st.error("⚠️ Target column not found. Please ensure your CSV has 'target' or 'GradeClass' column.")
                    with st.expander("📋 Expected Column Names"):
                        st.code("""
Required columns:
- StudentID, Age, Gender, Ethnicity, ParentalEducation
- StudyTimeWeekly, Absences, Tutoring, ParentalSupport
- Extracurricular, Sports, Music, Volunteering, GPA
- GradeClass (or 'target') - TARGET COLUMN
                        """)
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
                
                if target_col != 'target':
                    df = df.rename(columns={target_col: 'target'})
                
                X = df.drop('target', axis=1)
                y = df['target']
                
                model = load_model(model_key)
                scaler = load_scaler()
                
                if model is None:
                    st.error(f"❌ Model '{selected_display}' not found in saved_models/")
                    return
                
                scale_models = ['logistic_regression', 'knn', 'naive_bayes']
                if model_key in scale_models and scaler:
                    X_processed = scaler.transform(X)
                else:
                    X_processed = X
                
                # Get predictions with progress
                with st.spinner(f"🔄 Running {selected_display} predictions..."):
                    progress_bar = st.progress(0)
                    y_pred = model.predict(X_processed)
                    progress_bar.progress(50)
                    y_pred_proba = model.predict_proba(X_processed)
                    progress_bar.progress(100)
                    progress_bar.empty()
                
                # Calculate metrics
                metrics = calculate_metrics(y, y_pred, y_pred_proba)
                
                if metrics:
                    st.markdown(f'<p class="section-header">📈 {selected_display} - Results</p>', unsafe_allow_html=True)
                    
                    # Display metrics in cards with delta indicators
                    cols = st.columns(6)
                    metric_icons = ['🎯', '📊', '✅', '🔍', '⚖️', '📐']
                    metric_tooltips = {
                        'Accuracy': 'Overall prediction correctness',
                        'AUC': 'Area under ROC curve - class separation quality',
                        'Precision': 'True positives / (TP + FP) - false positive rate',
                        'Recall': 'True positives / (TP + FN) - false negative rate',
                        'F1': 'Harmonic mean of precision and recall',
                        'MCC': 'Matthews Correlation Coefficient - balanced metric'
                    }
                    
                    # Get best model for comparison
                    all_results = get_model_results()
                    best_metrics = {k: v.get('Accuracy', 0) for k, v in all_results.items()}
                    best_acc = max(best_metrics.values())
                    current_acc = metrics.get('Accuracy', 0)
                    
                    for col, (name, value), icon in zip(cols, metrics.items(), metric_icons):
                        with col:
                            # Show delta if accuracy
                            delta_val = None
                            if name == 'Accuracy' and current_acc < best_acc:
                                delta_val = f"-{(best_acc - current_acc)*100:.1f}%"
                            
                            st.metric(
                                label=name,
                                value=f"{value:.4f}",
                                delta=delta_val,
                                delta_color="inverse" if delta_val else "normal",
                                help=metric_tooltips.get(name, '')
                            )
                    
                    # Performance indicator
                    if current_acc >= 0.90:
                        st.success(f"🌟 Excellent performance! {current_acc*100:.2f}% accuracy")
                    elif current_acc >= 0.80:
                        st.info(f"✅ Good performance! {current_acc*100:.2f}% accuracy")
                    elif current_acc >= 0.70:
                        st.warning(f"⚠️ Moderate performance. {current_acc*100:.2f}% accuracy")
                    else:
                        st.error(f"❌ Low performance. {current_acc*100:.2f}% accuracy")
                    
                    # Confusion Matrix and Classification Report
                    st.markdown('<p class="section-header">🔍 Detailed Analysis</p>', unsafe_allow_html=True)
                    
                    tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📋 Classification Report", "📈 Prediction Confidence"])
                    
                    with tab1:
                        fig = plot_confusion_matrix(y, y_pred, f"Confusion Matrix - {selected_display}")
                        st.pyplot(fig)
                        
                        # Add accuracy per class
                        st.markdown("### 📊 Per-Class Accuracy")
                        class_acc = {}
                        for cls in np.unique(y):
                            mask = y == cls
                            if mask.sum() > 0:
                                class_acc[f'Grade {int(cls)}'] = accuracy_score(y[mask], y_pred[mask])
                        
                        if class_acc:
                            class_df = pd.DataFrame(list(class_acc.items()), columns=['Class', 'Accuracy'])
                            st.bar_chart(class_df.set_index('Class'), height=300)
                    
                    with tab2:
                        report = classification_report(y, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4), use_container_width=True)
                        
                        # Visualize per-class metrics
                        st.markdown("### 📊 Per-Class Metrics Visualization")
                        if 'macro avg' in report_df.index:
                            metrics_to_plot = ['precision', 'recall', 'f1-score']
                            class_rows = [idx for idx in report_df.index if idx not in ['accuracy', 'macro avg', 'weighted avg']]
                            
                            if class_rows:
                                plot_data = report_df.loc[class_rows, metrics_to_plot]
                                st.bar_chart(plot_data, height=350)
                    
                    with tab3:
                        st.markdown("### 📊 Prediction Confidence Distribution")
                        
                        # Show confidence scores
                        max_proba = np.max(y_pred_proba, axis=1)
                        confidence_df = pd.DataFrame({
                            'Confidence': max_proba,
                            'Predicted Class': [f'Grade {int(p)}' for p in y_pred],
                            'Actual Class': [f'Grade {int(a)}' for a in y]
                        })
                        
                        # Confidence histogram
                        st.markdown("**Confidence Score Distribution**")
                        st.bar_chart(pd.DataFrame({'Confidence': max_proba}), height=300)
                        
                        # Show samples with low confidence
                        low_conf_threshold = 0.5
                        low_conf_mask = max_proba < low_conf_threshold
                        if low_conf_mask.sum() > 0:
                            st.warning(f"⚠️ {low_conf_mask.sum()} predictions have confidence < {low_conf_threshold}")
                            with st.expander("View low confidence predictions"):
                                st.dataframe(confidence_df[low_conf_mask].head(20), use_container_width=True)
                        
                        # Confidence by class
                        st.markdown("**Average Confidence by Predicted Class**")
                        conf_by_class = confidence_df.groupby('Predicted Class')['Confidence'].mean().sort_values(ascending=False)
                        st.bar_chart(conf_by_class, height=300)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check your CSV file format.")
        
        else:
            # Welcome screen with enhanced UI
            st.markdown("""
            <div class="info-card animate-fade">
                <h3>👋 Welcome to the Student Performance Classifier</h3>
                <p>This application evaluates 6 machine learning models trained on the Student Performance dataset. 
                Upload your test data to see how well each model performs.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick stats
            all_results = get_model_results()
            best_model = max(all_results.items(), key=lambda x: x[1]['Accuracy'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🤖 Models", "6", "Trained & Ready")
            with col2:
                st.metric("🏆 Best Model", best_model[0], f"{best_model[1]['Accuracy']*100:.1f}%")
            with col3:
                st.metric("📊 Avg Accuracy", f"{np.mean([r['Accuracy'] for r in all_results.values()])*100:.1f}%", "Across all models")
            with col4:
                st.metric("📈 Best AUC", f"{max([r['AUC'] for r in all_results.values()]):.3f}", "XGBoost")
            
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
                
                ### 💡 Pro Tips
                - Use the **Compare Models** page to see all models side-by-side
                - Check **Prediction Confidence** tab to see model certainty
                - Download test data from sidebar if you need sample data
                """)
            
            with col2:
                st.markdown("### 🤖 Available Models")
                models = ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", 
                         "Naive Bayes", "Random Forest", "XGBoost"]
                for m in models:
                    acc = all_results.get(m, {}).get('Accuracy', 0)
                    st.markdown(f'<span class="model-badge">{m} ({acc*100:.1f}%)</span>', unsafe_allow_html=True)
                
                st.markdown("""
                
                ### 📊 Metrics Computed
                - **Accuracy**: Overall correctness
                - **AUC**: Class separation quality
                - **Precision**: True positives / (TP + FP)
                - **Recall**: True positives / (TP + FN)
                - **F1 Score**: Harmonic mean of precision & recall
                - **MCC**: Balanced correlation coefficient
                """)
            
            # Show sample comparison
            with st.expander("📊 Preview: Model Performance Comparison", expanded=False):
                preview_df = pd.DataFrame(all_results).T[['Accuracy', 'AUC', 'F1']].sort_values('Accuracy', ascending=False)
                st.dataframe(preview_df.style.highlight_max(axis=0, color='#22c55e').format("{:.4f}"), use_container_width=True)
    
    elif page == "📊 Compare Models":
        st.markdown('<p class="section-header">📊 Model Comparison Dashboard</p>', unsafe_allow_html=True)
        
        results = get_model_results()
        df_results = pd.DataFrame(results).T
        df_results.index.name = 'Model'
        
        # Interactive filter
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_metrics = st.multiselect(
                "Select Metrics to Compare",
                options=list(df_results.columns),
                default=list(df_results.columns),
                help="Choose which metrics to display in the comparison"
            )
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                options=['Accuracy', 'AUC', 'F1', 'MCC'],
                help="Sort models by this metric"
            )
        
        if selected_metrics:
            filtered_df = df_results[selected_metrics]
            
            # Summary table with interactive sorting
            st.markdown("### 📋 Performance Summary")
            sorted_df = filtered_df.sort_values(by=sort_by, ascending=False)
            
            st.dataframe(
                sorted_df.style.highlight_max(axis=0, color='#22c55e').format("{:.4f}"),
                use_container_width=True
            )
            
            # Best model indicator
            best_model = df_results['Accuracy'].idxmax()
            best_acc = df_results['Accuracy'].max()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏆 Best Model", best_model, f"{best_acc*100:.2f}%")
            with col2:
                st.metric("📊 Avg Accuracy", f"{df_results['Accuracy'].mean()*100:.2f}%", 
                        f"{df_results['Accuracy'].std()*100:.2f}% std")
            with col3:
                st.metric("📈 Best AUC", f"{df_results['AUC'].max():.4f}", 
                        df_results['AUC'].idxmax())
            
            # Interactive charts using Streamlit native
            st.markdown("### 📈 Interactive Comparison Charts")
            
            chart_tab1, chart_tab2 = st.tabs(["📊 Bar Charts", "📈 Line Chart"])
            
            with chart_tab1:
                # Create comparison chart
                for metric in selected_metrics[:3]:  # Show first 3 metrics
                    st.markdown(f"**{metric}**")
                    metric_data = sorted_df[metric].sort_values(ascending=True)
                    st.bar_chart(metric_data, height=250)
            
            with chart_tab2:
                # Line chart comparing all metrics
                st.markdown("**All Metrics Comparison**")
                st.line_chart(sorted_df[selected_metrics], height=400)
            
            # Visualization (matplotlib for detailed view)
            st.markdown("### 📈 Detailed Visual Comparison")
            fig = plot_metrics_comparison(results)
            st.pyplot(fig)
            
            # Rankings with interactive table
            st.markdown("### 🏅 Model Rankings by Metric")
            rankings = pd.DataFrame(index=df_results.index)
            for metric in df_results.columns:
                rankings[metric] = df_results[metric].rank(ascending=False).astype(int)
            rankings['Avg Rank'] = rankings.mean(axis=1).round(2)
            rankings = rankings.sort_values('Avg Rank')
            
            # Add color coding
            def highlight_ranks(val):
                if val == 1:
                    return 'background-color: #22c55e; color: white; font-weight: bold'
                elif val <= 3:
                    return 'background-color: #fbbf24; color: white'
                else:
                    return ''
            
            st.dataframe(
                rankings.style.applymap(highlight_ranks, subset=rankings.columns),
                use_container_width=True
            )
            
            # Model type grouping
            st.markdown("### 🎯 Performance by Model Type")
            model_types = {
                'Linear': ['Logistic Regression'],
                'Tree-based': ['Decision Tree'],
                'Instance-based': ['kNN'],
                'Probabilistic': ['Naive Bayes'],
                'Ensemble': ['Random Forest', 'XGBoost']
            }
            
            type_performance = {}
            for model_type, models in model_types.items():
                matching_models = [m for m in models if m in df_results.index]
                if matching_models:
                    type_performance[model_type] = df_results.loc[matching_models, 'Accuracy'].mean()
            
            if type_performance:
                type_df = pd.DataFrame(list(type_performance.items()), columns=['Model Type', 'Avg Accuracy'])
                type_df = type_df.sort_values('Avg Accuracy', ascending=True)
                st.bar_chart(type_df.set_index('Model Type'), height=300)
    
    elif page == "📖 Documentation":
        st.markdown('<p class="section-header">📖 Project Documentation</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Problem Statement", "📊 Dataset", "🤖 Models", "📖 How to Use"])
        
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
        
        with tab4:
            st.markdown("""
            ### 🚀 Quick Start Guide
            
            **Step 1: Select a Model**
            - Navigate to the **Model Evaluation** page
            - Choose a model from the sidebar dropdown (6 models available)
            - XGBoost typically performs best (92%+ accuracy)
            
            **Step 2: Upload Test Data**
            - Click "Upload CSV file" in the sidebar
            - Your CSV must include a target column (`GradeClass` or `target`)
            - Must have 14 features matching the training data format
            
            **Step 3: View Results**
            - See performance metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
            - Explore the Confusion Matrix tab for visual analysis
            - Check the Classification Report for per-class metrics
            - Review Prediction Confidence to see model certainty
            """)
            
            st.markdown("---")
            st.markdown("""
            ### 📂 Data Requirements
            
            Your test CSV file must have these columns:
            
            **Required Features (14):**
            - `StudentID`, `Age`, `Gender`, `Ethnicity`, `ParentalEducation`
            - `StudyTimeWeekly`, `Absences`, `Tutoring`, `ParentalSupport`
            - `Extracurricular`, `Sports`, `Music`, `Volunteering`, `GPA`
            
            **Target Column (required):**
            - `GradeClass` or `target` - Contains the actual class labels (0-4)
            
            **Example:**
            ```csv
            StudentID,Age,Gender,Ethnicity,ParentalEducation,StudyTimeWeekly,Absences,Tutoring,ParentalSupport,Extracurricular,Sports,Music,Volunteering,GPA,GradeClass
            1,18,Male,White,Some College,15,2,Yes,Yes,Yes,No,No,Yes,3.5,2
            ```
            """)
            
            st.markdown("---")
            st.markdown("""
            ### 📥 Download Files
            
            **From the Sidebar:**
            1. Navigate to **Model Evaluation** page
            2. Scroll to **"Download Files"** section
            3. Click **"Download test_data.csv"** - Get 500 sample test records
            4. Click **"Download train_models.ipynb"** - Get the training notebook
            
            **Test Data:**
            - 500 samples with same distributions as training data
            - Includes all 14 features (no target column)
            - Ready to use for predictions
            
            **Training Notebook:**
            - Complete Jupyter notebook with model training code
            - Step-by-step execution with markdown explanations
            - All 6 models included
            """)
            
            st.markdown("---")
            st.markdown("""
            ### 📊 Compare Models Page
            
            **Features:**
            - View all 6 models side-by-side
            - Select which metrics to compare
            - Sort by any metric (Accuracy, AUC, F1, MCC)
            - Interactive bar charts and line graphs
            - Model rankings by metric
            - Performance grouped by model type
            
            **How to Use:**
            1. Navigate to **"Compare Models"** from the sidebar
            2. Select metrics you want to compare (default: all)
            3. Choose sorting preference
            4. Explore charts and rankings
            """)
            
            st.markdown("---")
            st.markdown("""
            ### 💡 Tips & Best Practices
            
            **For Best Results:**
            - ✅ Use XGBoost or Random Forest for highest accuracy
            - ✅ Ensure your test data matches training data format
            - ✅ Check prediction confidence scores for reliability
            - ✅ Compare multiple models to find the best fit
            
            **Understanding Metrics:**
            - **Accuracy**: Overall prediction correctness
            - **AUC**: Class separation quality (higher is better)
            - **Precision**: Low false positive rate
            - **Recall**: Low false negative rate
            - **F1 Score**: Balance between precision and recall
            - **MCC**: Balanced metric for imbalanced classes
            
            **Troubleshooting:**
            - ❌ "Target column not found" → Ensure CSV has `GradeClass` or `target` column
            - ❌ "Model not found" → Run `python model/train_models.py` to train models
            - ❌ Low accuracy → Check if test data format matches training data
            """)
            
            st.markdown("---")
            st.markdown("""
            ### 🔧 Setup & Installation
            
            **Prerequisites:**
            - Python 3.8+ installed
            - pip package manager
            
            **Quick Setup:**
            ```bash
            # Create virtual environment
            python -m venv venv
            source venv/bin/activate  # Mac/Linux
            # OR
            venv\\Scripts\\activate  # Windows
            
            # Install dependencies
            pip install -r requirements.txt
            
            # Train models (optional - pre-trained models included)
            python model/train_models.py
            
            # Run the app
            streamlit run app.py
            ```
            
            For detailed setup instructions, see `QUICK_START.md` in the project root.
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
