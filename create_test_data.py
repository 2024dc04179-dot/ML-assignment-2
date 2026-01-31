import pandas as pd
import numpy as np
import os

# Set random seed
np.random.seed(42)

# Get the script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

# Read training data
train_path = os.path.join(data_dir, 'Student_performance_data.csv')
print(f"Loading training data from {train_path}...")
train_df = pd.read_csv(train_path)
print(f"Training data shape: {train_df.shape}")

# Remove target column
train_features = train_df.drop('GradeClass', axis=1)
n_samples = 500

# Generate test data
print(f"Generating {n_samples} test samples...")
test_data = {}

# StudentID: Sequential from max + 1
start_id = train_df['StudentID'].max() + 1
test_data['StudentID'] = range(start_id, start_id + n_samples)

# Age: Sample from same values
test_data['Age'] = np.random.choice(train_df['Age'].values, size=n_samples, replace=True)

# Gender: Maintain proportion
gender_probs = train_df['Gender'].value_counts(normalize=True).sort_index()
test_data['Gender'] = np.random.choice(gender_probs.index.values, size=n_samples, p=gender_probs.values, replace=True)

# Ethnicity: Maintain distribution
eth_probs = train_df['Ethnicity'].value_counts(normalize=True).sort_index()
test_data['Ethnicity'] = np.random.choice(eth_probs.index.values, size=n_samples, p=eth_probs.values, replace=True)

# ParentalEducation: Maintain distribution
pe_probs = train_df['ParentalEducation'].value_counts(normalize=True).sort_index()
test_data['ParentalEducation'] = np.random.choice(pe_probs.index.values, size=n_samples, p=pe_probs.values, replace=True)

# StudyTimeWeekly: Normal distribution with same mean/std
mean_st = train_df['StudyTimeWeekly'].mean()
std_st = train_df['StudyTimeWeekly'].std()
test_data['StudyTimeWeekly'] = np.maximum(0, np.random.normal(mean_st, std_st, n_samples))

# Absences: Sample from same values
test_data['Absences'] = np.random.choice(train_df['Absences'].values, size=n_samples, replace=True)

# Tutoring: Maintain proportion
tut_probs = train_df['Tutoring'].value_counts(normalize=True).sort_index()
test_data['Tutoring'] = np.random.choice(tut_probs.index.values, size=n_samples, p=tut_probs.values, replace=True)

# ParentalSupport: Maintain distribution
ps_probs = train_df['ParentalSupport'].value_counts(normalize=True).sort_index()
test_data['ParentalSupport'] = np.random.choice(ps_probs.index.values, size=n_samples, p=ps_probs.values, replace=True)

# Extracurricular: Maintain proportion
ec_probs = train_df['Extracurricular'].value_counts(normalize=True).sort_index()
test_data['Extracurricular'] = np.random.choice(ec_probs.index.values, size=n_samples, p=ec_probs.values, replace=True)

# Sports: Maintain proportion
sports_probs = train_df['Sports'].value_counts(normalize=True).sort_index()
test_data['Sports'] = np.random.choice(sports_probs.index.values, size=n_samples, p=sports_probs.values, replace=True)

# Music: Maintain proportion
music_probs = train_df['Music'].value_counts(normalize=True).sort_index()
test_data['Music'] = np.random.choice(music_probs.index.values, size=n_samples, p=music_probs.values, replace=True)

# Volunteering: Maintain proportion
vol_probs = train_df['Volunteering'].value_counts(normalize=True).sort_index()
test_data['Volunteering'] = np.random.choice(vol_probs.index.values, size=n_samples, p=vol_probs.values, replace=True)

# GPA: Normal distribution clipped to 0-4
mean_gpa = train_df['GPA'].mean()
std_gpa = train_df['GPA'].std()
test_data['GPA'] = np.clip(np.random.normal(mean_gpa, std_gpa, n_samples), 0, 4.0)

# Create DataFrame with same column order
test_df = pd.DataFrame(test_data)
test_df = test_df[train_features.columns]

# Save to CSV
test_data_path = os.path.join(data_dir, 'test_data.csv')
test_df.to_csv(test_data_path, index=False)
print(f"\n[SUCCESS] Test data saved to {test_data_path}")
print(f"   Shape: {test_df.shape}")
print(f"   Columns: {list(test_df.columns)}")
print(f"\n[INFO] This file can be committed to GitHub")

