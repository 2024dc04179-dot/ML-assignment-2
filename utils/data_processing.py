"""Data processing utilities"""

import pandas as pd
import numpy as np
import os


def generate_test_data_if_needed():
    """Generate test data if it doesn't exist"""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        return False


