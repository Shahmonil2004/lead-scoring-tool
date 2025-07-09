import os
import pandas as pd
os.chdir('d:/E_Drive/MONIL/STUDIES/python/lead optimizer')
# Print the current working directory
print("Current Working Directory:", os.getcwd())
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('SampleData.csv', encoding='ISO-8859-1')

# Define the required columns for lead validation and scoring
required_columns = [
    'Prospect ID', 'Company', 'Lead Number', 'Mobile Number', 'Lead Source', 'Lead Origin',
    'Country', 'Job Title', 'Lead Stage', 'Engagement Score', 'Website', 
    'TotalVisits', 'Page Views Per Visit', 'Average Time Per Visit', 'Industry', 
    'Source Campaign', 'Last Activity', 'Lead Type'
]

# Keep only the required columns
data_cleaned = data[required_columns]

# Check for missing values in the selected columns
print("\nMissing values in each column:")
print(data_cleaned.isnull().sum())

# Handling missing values: We do not fill missing values for validation columns.
# For real-time validation, missing values will remain as NaN.

# Encode categorical columns (if needed)
label_encoder = LabelEncoder()
for col in data_cleaned.select_dtypes(include=['object']).columns:
    data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col].astype(str))  # Convert to string

# Display the cleaned data
print("\nCleaned Data (first 5 rows):")
print(data_cleaned.head())

# Save the cleaned data to a new CSV file
data_cleaned.to_csv('Cleaned_Data.csv', index=False)

print("\nData preprocessing complete and cleaned data saved as 'Cleaned_Data.csv'.")
