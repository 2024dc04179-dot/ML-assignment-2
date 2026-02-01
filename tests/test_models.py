"""
Unit Tests for ML Assignment 2
Student Performance Classification Models

Author: Abhishek Anand (2024DC04179)
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import accuracy_score, f1_score


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def project_root():
    """Get project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def saved_models_dir(project_root):
    """Get saved models directory"""
    return os.path.join(project_root, 'saved_models')


@pytest.fixture
def data_dir(project_root):
    """Get data directory"""
    return os.path.join(project_root, 'data')


@pytest.fixture
def sample_data(data_dir):
    """Load sample test data"""
    csv_path = os.path.join(data_dir, 'Student_performance_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.sample(n=min(100, len(df)), random_state=42)
    return None


@pytest.fixture
def scaler(saved_models_dir):
    """Load scaler"""
    scaler_path = os.path.join(saved_models_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None


# ============================================================================
# Test: Model Files Exist
# ============================================================================

class TestModelFiles:
    """Test that all required model files exist"""
    
    MODEL_FILES = [
        'logistic_regression.pkl',
        'decision_tree.pkl',
        'knn.pkl',
        'naive_bayes.pkl',
        'random_forest.pkl',
        'xgboost.pkl',
        'scaler.pkl'
    ]
    
    def test_saved_models_directory_exists(self, saved_models_dir):
        """Check saved_models directory exists"""
        assert os.path.isdir(saved_models_dir), "saved_models/ directory not found"
    
    @pytest.mark.parametrize("model_file", MODEL_FILES)
    def test_model_file_exists(self, saved_models_dir, model_file):
        """Check each model file exists"""
        model_path = os.path.join(saved_models_dir, model_file)
        assert os.path.exists(model_path), f"{model_file} not found in saved_models/"
    
    @pytest.mark.parametrize("model_file", MODEL_FILES)
    def test_model_file_not_empty(self, saved_models_dir, model_file):
        """Check model files are not empty"""
        model_path = os.path.join(saved_models_dir, model_file)
        if os.path.exists(model_path):
            assert os.path.getsize(model_path) > 0, f"{model_file} is empty"


# ============================================================================
# Test: Model Loading
# ============================================================================

class TestModelLoading:
    """Test that models can be loaded correctly"""
    
    MODELS = [
        'logistic_regression',
        'decision_tree',
        'knn',
        'naive_bayes',
        'random_forest',
        'xgboost'
    ]
    
    @pytest.mark.parametrize("model_name", MODELS)
    def test_model_loads_without_error(self, saved_models_dir, model_name):
        """Test model loads successfully"""
        model_path = os.path.join(saved_models_dir, f'{model_name}.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            assert model is not None, f"Failed to load {model_name}"
    
    @pytest.mark.parametrize("model_name", MODELS)
    def test_model_has_predict_method(self, saved_models_dir, model_name):
        """Test model has predict method"""
        model_path = os.path.join(saved_models_dir, f'{model_name}.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            assert hasattr(model, 'predict'), f"{model_name} missing predict method"
    
    @pytest.mark.parametrize("model_name", MODELS)
    def test_model_has_predict_proba_method(self, saved_models_dir, model_name):
        """Test model has predict_proba method"""
        model_path = os.path.join(saved_models_dir, f'{model_name}.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            assert hasattr(model, 'predict_proba'), f"{model_name} missing predict_proba method"
    
    def test_scaler_loads_correctly(self, saved_models_dir):
        """Test scaler loads and has transform method"""
        scaler_path = os.path.join(saved_models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            assert hasattr(scaler, 'transform'), "Scaler missing transform method"


# ============================================================================
# Test: Dataset
# ============================================================================

class TestDataset:
    """Test dataset requirements"""
    
    def test_dataset_exists(self, data_dir):
        """Check dataset file exists"""
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        assert len(csv_files) > 0, "No CSV files found in data/"
    
    def test_dataset_minimum_instances(self, data_dir):
        """Check dataset has minimum 500 instances (requirement)"""
        csv_path = os.path.join(data_dir, 'Student_performance_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            assert len(df) >= 500, f"Dataset has {len(df)} instances, minimum is 500"
    
    def test_dataset_minimum_features(self, data_dir):
        """Check dataset has minimum 12 features (requirement)"""
        csv_path = os.path.join(data_dir, 'Student_performance_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            n_features = len(df.columns) - 1  # Exclude target
            assert n_features >= 12, f"Dataset has {n_features} features, minimum is 12"
    
    def test_dataset_has_target_column(self, data_dir):
        """Check dataset has target column"""
        csv_path = os.path.join(data_dir, 'Student_performance_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            target_cols = ['GradeClass', 'target', 'Target']
            has_target = any(col in df.columns for col in target_cols)
            assert has_target, "Dataset missing target column (GradeClass/target)"
    
    def test_dataset_no_missing_values(self, data_dir):
        """Check dataset has no missing values"""
        csv_path = os.path.join(data_dir, 'Student_performance_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
    
    def test_test_data_generation_works(self, data_dir):
        """Check test data can be generated (if not exists)"""
        test_data_path = os.path.join(data_dir, 'test_data.csv')
        # Import the function to test
        try:
            from utils.data_processing import generate_test_data_if_needed
            result = generate_test_data_if_needed()
            # Should return True if file exists or was created successfully
            assert result is True or os.path.exists(test_data_path), "Test data generation failed"
        except ImportError:
            pytest.skip("Cannot import generate_test_data_if_needed")


# ============================================================================
# Test: Model Predictions
# ============================================================================

class TestModelPredictions:
    """Test model predictions work correctly"""
    
    def test_logistic_regression_prediction(self, saved_models_dir, sample_data, scaler):
        """Test Logistic Regression makes valid predictions"""
        if sample_data is None:
            pytest.skip("Sample data not available")
        
        model_path = os.path.join(saved_models_dir, 'logistic_regression.pkl')
        if not os.path.exists(model_path):
            pytest.skip("Model file not found")
        
        model = joblib.load(model_path)
        X = sample_data.drop('GradeClass', axis=1)
        
        if scaler:
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X)
        
        assert len(predictions) == len(X), "Prediction count mismatch"
        assert all(p in [0, 1, 2, 3, 4] for p in predictions), "Invalid prediction values"
    
    def test_decision_tree_prediction(self, saved_models_dir, sample_data):
        """Test Decision Tree makes valid predictions"""
        if sample_data is None:
            pytest.skip("Sample data not available")
        
        model_path = os.path.join(saved_models_dir, 'decision_tree.pkl')
        if not os.path.exists(model_path):
            pytest.skip("Model file not found")
        
        model = joblib.load(model_path)
        X = sample_data.drop('GradeClass', axis=1)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X), "Prediction count mismatch"
    
    def test_random_forest_prediction(self, saved_models_dir, sample_data):
        """Test Random Forest makes valid predictions"""
        if sample_data is None:
            pytest.skip("Sample data not available")
        
        model_path = os.path.join(saved_models_dir, 'random_forest.pkl')
        if not os.path.exists(model_path):
            pytest.skip("Model file not found")
        
        model = joblib.load(model_path)
        X = sample_data.drop('GradeClass', axis=1)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X), "Prediction count mismatch"
    
    def test_xgboost_prediction(self, saved_models_dir, sample_data):
        """Test XGBoost makes valid predictions"""
        if sample_data is None:
            pytest.skip("Sample data not available")
        
        model_path = os.path.join(saved_models_dir, 'xgboost.pkl')
        if not os.path.exists(model_path):
            pytest.skip("Model file not found")
        
        model = joblib.load(model_path)
        X = sample_data.drop('GradeClass', axis=1)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X), "Prediction count mismatch"


# ============================================================================
# Test: Model Performance
# ============================================================================

class TestModelPerformance:
    """Test models meet minimum performance thresholds"""
    
    def test_best_model_accuracy_above_threshold(self, saved_models_dir, sample_data):
        """Test best model achieves reasonable accuracy"""
        if sample_data is None:
            pytest.skip("Sample data not available")
        
        model_path = os.path.join(saved_models_dir, 'xgboost.pkl')
        if not os.path.exists(model_path):
            pytest.skip("XGBoost model not found")
        
        model = joblib.load(model_path)
        X = sample_data.drop('GradeClass', axis=1)
        y = sample_data['GradeClass']
        
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        # XGBoost should achieve at least 80% on sample
        assert accuracy >= 0.80, f"XGBoost accuracy {accuracy:.4f} below 80% threshold"
    
    def test_probability_outputs_valid(self, saved_models_dir, sample_data):
        """Test probability outputs sum to 1"""
        if sample_data is None:
            pytest.skip("Sample data not available")
        
        model_path = os.path.join(saved_models_dir, 'random_forest.pkl')
        if not os.path.exists(model_path):
            pytest.skip("Random Forest model not found")
        
        model = joblib.load(model_path)
        X = sample_data.drop('GradeClass', axis=1)
        
        proba = model.predict_proba(X)
        
        # Check probabilities sum to 1 (with tolerance)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities don't sum to 1"


# ============================================================================
# Test: Requirements File
# ============================================================================

class TestRequirements:
    """Test requirements.txt is properly configured"""
    
    def test_requirements_file_exists(self, project_root):
        """Check requirements.txt exists"""
        req_path = os.path.join(project_root, 'requirements.txt')
        assert os.path.exists(req_path), "requirements.txt not found"
    
    def test_required_packages_listed(self, project_root):
        """Check essential packages are in requirements"""
        req_path = os.path.join(project_root, 'requirements.txt')
        if not os.path.exists(req_path):
            pytest.skip("requirements.txt not found")
        
        with open(req_path, 'r') as f:
            content = f.read().lower()
        
        required = ['streamlit', 'scikit-learn', 'pandas', 'numpy', 'xgboost']
        for pkg in required:
            assert pkg in content, f"{pkg} not found in requirements.txt"


# ============================================================================
# Test: App File
# ============================================================================

class TestTrainingFiles:
    """Test model training files exist"""
    
    def test_train_models_py_exists(self, project_root):
        """Check train_models.py exists"""
        train_script = os.path.join(project_root, 'model', 'train_models.py')
        assert os.path.exists(train_script), "model/train_models.py not found"
    
    def test_train_models_ipynb_exists(self, project_root):
        """Check train_models.ipynb exists"""
        train_notebook = os.path.join(project_root, 'model', 'train_models.ipynb')
        assert os.path.exists(train_notebook), "model/train_models.ipynb not found"
    
    def test_test_data_csv_exists(self, data_dir):
        """Check test_data.csv exists (for download feature)"""
        test_data_path = os.path.join(data_dir, 'test_data.csv')
        # Test data might be auto-generated, so this is optional
        if not os.path.exists(test_data_path):
            pytest.skip("test_data.csv not found (will be auto-generated by app)")


class TestModularStructure:
    """Test modular code structure"""
    
    def test_utils_directory_exists(self, project_root):
        """Check utils/ directory exists"""
        utils_dir = os.path.join(project_root, 'utils')
        assert os.path.isdir(utils_dir), "utils/ directory not found"
    
    def test_utils_modules_exist(self, project_root):
        """Check required utils modules exist"""
        utils_dir = os.path.join(project_root, 'utils')
        required_modules = ['model_loader.py', 'data_processing.py', 'metrics.py', 'visualizations.py']
        for module in required_modules:
            module_path = os.path.join(utils_dir, module)
            assert os.path.exists(module_path), f"utils/{module} not found"
    
    def test_styles_directory_exists(self, project_root):
        """Check styles/ directory exists"""
        styles_dir = os.path.join(project_root, 'styles')
        assert os.path.isdir(styles_dir), "styles/ directory not found"
    
    def test_styles_modules_exist(self, project_root):
        """Check required styles modules exist"""
        styles_dir = os.path.join(project_root, 'styles')
        required_modules = ['css.py', 'javascript.py']
        for module in required_modules:
            module_path = os.path.join(styles_dir, module)
            assert os.path.exists(module_path), f"styles/{module} not found"
    
    def test_components_directory_exists(self, project_root):
        """Check components/ directory exists"""
        components_dir = os.path.join(project_root, 'components')
        assert os.path.isdir(components_dir), "components/ directory not found"


class TestAppFile:
    """Test app.py configuration"""
    
    def test_app_file_exists(self, project_root):
        """Check app.py exists"""
        app_path = os.path.join(project_root, 'app.py')
        assert os.path.exists(app_path), "app.py not found"
    
    def test_app_imports_streamlit(self, project_root):
        """Check app imports streamlit"""
        app_path = os.path.join(project_root, 'app.py')
        if not os.path.exists(app_path):
            pytest.skip("app.py not found")
        
        with open(app_path, 'r') as f:
            content = f.read()
        
        assert 'import streamlit' in content, "app.py doesn't import streamlit"
    
    def test_app_has_main_function(self, project_root):
        """Check app has main function"""
        app_path = os.path.join(project_root, 'app.py')
        if not os.path.exists(app_path):
            pytest.skip("app.py not found")
        
        with open(app_path, 'r') as f:
            content = f.read()
        
        assert 'def main' in content, "app.py doesn't have main function"
    
    def test_app_imports_modular_components(self, project_root):
        """Check app imports from utils, styles modules"""
        app_path = os.path.join(project_root, 'app.py')
        if not os.path.exists(app_path):
            pytest.skip("app.py not found")
        
        with open(app_path, 'r') as f:
            content = f.read()
        
        assert 'from utils' in content or 'import utils' in content, "app.py doesn't import utils module"
        assert 'from styles' in content or 'import styles' in content, "app.py doesn't import styles module"


# ============================================================================
# Test: README
# ============================================================================

class TestReadme:
    """Test README.md requirements"""
    
    def test_readme_exists(self, project_root):
        """Check README.md exists"""
        readme_path = os.path.join(project_root, 'README.md')
        assert os.path.exists(readme_path), "README.md not found"
    
    def test_readme_has_problem_statement(self, project_root):
        """Check README has problem statement section"""
        readme_path = os.path.join(project_root, 'README.md')
        if not os.path.exists(readme_path):
            pytest.skip("README.md not found")
        
        with open(readme_path, 'r') as f:
            content = f.read().lower()
        
        assert 'problem statement' in content, "README missing problem statement"
    
    def test_readme_has_dataset_description(self, project_root):
        """Check README has dataset description"""
        readme_path = os.path.join(project_root, 'README.md')
        if not os.path.exists(readme_path):
            pytest.skip("README.md not found")
        
        with open(readme_path, 'r') as f:
            content = f.read().lower()
        
        assert 'dataset' in content, "README missing dataset description"
    
    def test_readme_has_comparison_table(self, project_root):
        """Check README has model comparison table"""
        readme_path = os.path.join(project_root, 'README.md')
        if not os.path.exists(readme_path):
            pytest.skip("README.md not found")
        
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Check for table format
        assert '|' in content, "README missing comparison table"
        assert 'Accuracy' in content, "README missing Accuracy metric"
        assert 'AUC' in content, "README missing AUC metric"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
