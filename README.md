# Lead Scoring Tool

This is a real-time lead scoring tool that predicts lead scores based on various lead details such as engagement, visits, time spent, and lead type. The tool uses a pre-trained machine learning model to classify leads and assign a score. The scores help businesses prioritize their leads based on the likelihood of conversion, improving sales outreach and decision-making processes.

## Installation

To set up and run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Shahmonil2004/lead-scoring-tool.git
Navigate to the project directory:

cd lead-scoring-tool

Create a virtual environment (optional but recommended):

python -m venv lead_optimizer_env

Activate the virtual environment:

    On Windows:

lead_optimizer_env\Scripts\activate

On macOS/Linux:

    source lead_optimizer_env/bin/activate

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit app:

    streamlit run lead_scoring_app.py

    The app will be available at http://localhost:8501 in your web browser.

Usage

    Input lead details into the input fields (e.g., Engagement Score, Total Visits, etc.).

    Click the "Submit" button to get the predicted lead score.

    The app will display a table of all leads with their predicted scores, ranked in descending order of score.

    You can also view visualizations like:

        Lead score distribution (Histogram)

        Engagement score vs. lead score (Scatter plot)

Technologies Used

    Streamlit: For creating the web app interface.

    pandas: For data manipulation and storage.

    scikit-learn: For machine learning model training and prediction.

    matplotlib & seaborn: For data visualization.

    joblib: For saving and loading the pre-trained model.

Features

    Real-time lead scoring prediction based on input parameters.

    Data visualization of lead score distribution and relationships.

    Table of leads ranked by score for easy analysis.

    Export option for saving leads data as CSV.

Contributing

If you'd like to contribute to this project, please follow these steps:

    Fork the repository.

    Create a new branch (git checkout -b feature-branch).

    Make your changes and commit them (git commit -am 'Add new feature').

    Push to the branch (git push origin feature-branch).

    Open a pull request to merge your changes.

License

This project is licensed under the MIT License - see the LICENSE file for details.
Contact Information

Monil Shah
Email: shah2004monil@gmail.com
GitHub: https://github.com/Shahmonil2004


