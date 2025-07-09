import pandas as pd
import re
import os
os.chdir('d:/E_Drive/MONIL/STUDIES/python/lead optimizer')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read data (replace with your actual dataset)
data = pd.read_csv('Cleaned_Data.csv')

# Fill missing values with zero
data.fillna(0, inplace=True)

# Define the scoring formula (normalize the features and generate a score between 0 and 100)
def calculate_lead_score(row):
    # Example formula, can be adjusted to your needs
    score = (
        0.3 * row['Engagement Score'] +
        0.2 * row['TotalVisits'] +
        0.15 * row['Page Views Per Visit'] +
        0.15 * row['Average Time Per Visit'] +
        0.1 * row['Lead Stage'] +
        0.1 * row['Lead Type']
    )

    # Normalize score to the range 0-100
    score = max(0, min(100, score))  # Ensure it's between 0 and 100
    return score

# Apply the scoring function to the dataset and store the lead scores as a new column
data['Lead Score'] = data.apply(calculate_lead_score, axis=1)

# Save the updated data with lead scores to a new CSV file
data.to_csv('Leads_with_Predicted_Scores.csv', index=False)

# Select features and target variable (lead score)
X = data[['Engagement Score', 'TotalVisits', 'Page Views Per Visit', 'Average Time Per Visit', 'Lead Stage', 'Lead Type']]
y = data['Lead Score']  # This is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the trained model for future use
import joblib
joblib.dump(model, 'lead_scoring_model.pkl')

# Predict lead scores for new data (example)
new_data = pd.DataFrame({
    'Engagement Score': [5.2, 3.1],
    'TotalVisits': [100, 150],
    'Page Views Per Visit': [3.5, 4.2],
    'Average Time Per Visit': [5.1, 3.2],
    'Lead Stage': [2, 1],
    'Lead Type': [2, 1]
})

# Predict lead scores for the new data
new_data['Predicted Lead Score'] = model.predict(new_data)

# Show the predicted lead scores
print(new_data)

