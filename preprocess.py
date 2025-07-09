import os
import pandas as pd
os.chdir('d:/E_Drive/MONIL/STUDIES/python/lead optimizer')

print("Current Working Directory:", os.getcwd())
import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('SampleData.csv', encoding='ISO-8859-1')


required_columns = [
    'Prospect ID', 'Company', 'Lead Number', 'Mobile Number', 'Lead Source', 'Lead Origin',
    'Country', 'Job Title', 'Lead Stage', 'Engagement Score', 'Website', 
    'TotalVisits', 'Page Views Per Visit', 'Average Time Per Visit', 'Industry', 
    'Source Campaign', 'Last Activity', 'Lead Type'
]


data_cleaned = data[required_columns]


print("\nMissing values in each column:")
print(data_cleaned.isnull().sum())





label_encoder = LabelEncoder()
for col in data_cleaned.select_dtypes(include=['object']).columns:
    data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col].astype(str))  
print("\nCleaned Data (first 5 rows):")
print(data_cleaned.head())


data_cleaned.to_csv('Cleaned_Data.csv', index=False)

print("\nData preprocessing complete and cleaned data saved as 'Cleaned_Data.csv'.")
