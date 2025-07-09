import os
os.chdir('d:/E_Drive/MONIL/STUDIES/python/lead optimizer')
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model using joblib
model = joblib.load('lead_scoring_model.pkl')  # Replace with the actual model path

# Streamlit UI setup
st.set_page_config(page_title="Lead Scoring Tool", page_icon="📊", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            color: #2A73CC;
            font-weight: bold;
            text-align: center;
        }
        .subheader {
            font-size: 30px;
            color: #4CAF50;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .sidebar {
            background-color: #F4F4F4;
            padding: 10px;
            border-radius: 10px;
        }
        .input-section {
            background-color: #E8F1F2;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .input-box {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
        }
        .submit-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .submit-btn:hover {
            background-color: #45a049;
        }
        .csv-btn {
            background-color: #2A73CC;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .csv-btn:hover {
            background-color: #0066CC;
        }
        .lead-table th {
            background-color: #2A73CC;
            color: white;
            padding: 10px;
            text-align: center;
        }
        .lead-table td {
            padding: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='title'>Real-Time Lead Scoring Tool</h1>", unsafe_allow_html=True)
st.write("Input lead details below to get the real-time lead score.")

# User input fields for lead data (Only necessary features used in model training)
with st.form(key='lead_form'):
    # Styling for input section
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    lead_data = {
        "Engagement Score": st.slider("Engagement Score", min_value=0, max_value=10, value=5),
        "TotalVisits": st.number_input("Total Visits", min_value=1, value=5),
        "Page Views Per Visit": st.number_input("Page Views Per Visit", min_value=1, value=3),
        "Average Time Per Visit": st.number_input("Average Time Per Visit (seconds)", min_value=1, value=120),
        "Lead Stage": st.selectbox("Lead Stage", ["New", "Contacted", "Qualified", "Closed"]),
        "Lead Type": st.selectbox("Lead Type", ["B2B", "B2C"]),
    }
    
    # Submit button for form
    submit_button = st.form_submit_button(label="Submit Lead", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)  # End input-section div

# Check if the submit button is pressed
if submit_button:
    # Convert user input into a DataFrame that can be passed into the model
    user_input = pd.DataFrame([lead_data])

    # Function to preprocess inputs before passing into the model
    def preprocess_input(data):
        # Convert categorical columns to numerical
        data['Lead Stage'] = data['Lead Stage'].map({"New": 1, "Contacted": 2, "Qualified": 3, "Closed": 4})
        data['Lead Type'] = data['Lead Type'].map({"B2B": 1, "B2C": 2})

        # Ensure all other fields are numeric or filled with a default value if missing
        data.fillna(0, inplace=True)
        
        # Reorder the columns to match the model's training data
        columns_order = ['Engagement Score', 'TotalVisits', 'Page Views Per Visit', 'Average Time Per Visit', 'Lead Stage', 'Lead Type']
        data = data[columns_order]
        
        return data

    # Preprocess user input
    processed_input = preprocess_input(user_input)

    # Make predictions using the trained model
    predicted_score = model.predict(processed_input)

    # Show the predicted lead score
    st.write(f"Predicted Lead Score: {predicted_score[0]:.2f}")

    # Store the predicted lead score in the dataframe and display it
    lead_data["Lead Score"] = predicted_score[0]
    lead_data_df = pd.DataFrame([lead_data])

    # Now, let's store the entered data in the session state for future predictions
    if "leads_df" not in st.session_state:
        st.session_state['leads_df'] = pd.DataFrame(columns=["Engagement Score", "TotalVisits", "Page Views Per Visit", 
                                                            "Average Time Per Visit", "Lead Stage", "Lead Type", "Lead Score"])

    # Append new lead data to session state
    st.session_state['leads_df'] = pd.concat([st.session_state['leads_df'], lead_data_df], ignore_index=True)

    # Display all entered leads ranked by Lead Score
    st.write("### Leads Ranked by Lead Score")
    st.dataframe(st.session_state['leads_df'].sort_values(by="Lead Score", ascending=False), use_container_width=True)

    # Data Visualization: Lead Score Distribution
    st.write("### Lead Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(st.session_state['leads_df']["Lead Score"], kde=True, ax=ax)
    ax.set_title("Distribution of Lead Scores")
    ax.set_xlabel("Lead Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Data Visualization: Engagement Score vs Lead Score
    st.write("### Engagement Score vs Lead Score")
    fig, ax = plt.subplots()
    sns.scatterplot(x=st.session_state['leads_df']["Engagement Score"], y=st.session_state['leads_df']["Lead Score"], ax=ax)
    ax.set_title("Engagement Score vs Lead Score")
    ax.set_xlabel("Engagement Score")
    ax.set_ylabel("Lead Score")
    st.pyplot(fig)

    # Additional: Save Data as CSV
    if st.button("Save Leads Data as CSV"):
        st.session_state['leads_df'].to_csv("leads_data.csv", index=False)
        st.write("Data saved as leads_data.csv")
