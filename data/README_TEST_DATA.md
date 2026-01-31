# Test Data

The `test_data.csv` file contains 500 test samples with the same feature distributions as the training data, but **without the target column** (GradeClass).

## Generating Test Data

The test data can be generated in two ways:

1. **Automatically via Streamlit App**: When you run the Streamlit app, it will automatically generate `test_data.csv` if it doesn't exist.

2. **Manually via Script**: Run one of these scripts:
   ```bash
   python create_test_data.py
   # or
   python generate_test_data.py
   ```

## File Location

- **Path**: `data/test_data.csv`
- **Status**: This file is tracked by Git and can be committed to GitHub
- **Purpose**: Used for testing model implementations without the target column

## Features

The test data includes all 14 features from the training data:
- StudentID
- Age
- Gender
- Ethnicity
- ParentalEducation
- StudyTimeWeekly
- Absences
- Tutoring
- ParentalSupport
- Extracurricular
- Sports
- Music
- Volunteering
- GPA

**Note**: The target column (GradeClass) is excluded as this is test data for model evaluation.

