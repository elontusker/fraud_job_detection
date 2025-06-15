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

✅ Model Performance Summary
The model achieved an overall accuracy of 88% on the test dataset of 2,861 samples.

Class 0 (likely the majority or "non-fraud" class):

Precision: 1.00 → Every predicted class 0 was actually class 0.

Recall: 0.88 → 88% of actual class 0 instances were correctly identified.

F1-score: 0.93 → High harmonic mean of precision and recall indicates strong performance on this class.

Class 1 (likely the minority or "fraud" class):

Precision: 0.29 → Only 29% of predicted fraud cases were truly fraud.

Recall: 0.96 → The model correctly identified 96% of actual fraud cases.

F1-score: 0.45 → Indicates trade-off between low precision and high recall.

📊 Macro and Weighted Averages:
Macro average:

Precision: 0.64, Recall: 0.92, F1-score: 0.69

This gives equal weight to both classes and highlights imbalance in performance across classes.

Weighted average:

Precision: 0.96, Recall: 0.88, F1-score: 0.91

This accounts for the class imbalance, reflecting the model’s high performance on the majority class.

🔍 Insights:
The model is highly sensitive (recall) for detecting fraud (Class 1), which is often desirable in fraud detection use cases.

However, precision for fraud is low, meaning many false positives — the model flags many non-fraud cases as fraud.

This trade-off suggests the model is tuned for recall over precision, potentially acceptable depending on the business goal (e.g., flagging all potential fraud for further review).
