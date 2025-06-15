# Job Posting Fraud Detection System

A machine learning system to detect fraudulent job postings, with a Flask-based dashboard for interactive analysis.

## Features

- Binary classification of job postings as genuine or fraudulent
- CSV file upload and processing
- Interactive dashboard with visual insights:
  - Fraud probability distribution
  - Percentage of fraudulent vs genuine jobs
  - Top 10 most suspicious postings
  - Detailed results table
- Downloadable results in CSV format

## Technologies Used

- Python 3
- Flask (web framework)
- scikit-learn (machine learning)
- pandas (data processing)
- matplotlib/seaborn (visualizations)
- Bootstrap (frontend styling)

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraud_job_detection.git
   cd fraud_job_detection

## Running the System

1. Save your training data as `data/training_data.csv`
2. Run `python model.py` to train the model
3. Run `python app.py` to start the Flask server
4. Access the dashboard at http://localhost:5000
5. Upload a CSV file with job postings to analyze

## Key Features Implemented

1. Binary classifier (Random Forest) trained on your data
2. CSV upload and prediction functionality
3. Dashboard with:
   - Results table
   - Fraud probability histogram
   - Pie chart of fake vs real jobs
   - Top 10 most suspicious listings
4. Complete setup instructions in README
5. Modular code structure

The system handles class imbalance through class weighting in the Random Forest classifier and evaluates performance using F1-score as requested. The dashboard provides clear visual insights into the analysis results.
