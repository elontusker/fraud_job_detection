import os
import pandas as pd
import joblib
from model import train_model, preprocess_data
from datetime import datetime

# Configuration
MODEL_PATH = 'model.joblib'
DATA_DIR = 'data'
TRAINING_DATA = os.path.join(DATA_DIR, 'training_data.csv')
NEW_DATA_DIR = os.path.join(DATA_DIR, 'new_data')
ARCHIVE_DIR = os.path.join(DATA_DIR, 'archive')

def check_for_new_data():
    """Check for new data files to incorporate"""
    new_files = []
    for filename in os.listdir(NEW_DATA_DIR):
        if filename.endswith('.xlsx') or filename.endswith('.csv'):
            new_files.append(os.path.join(NEW_DATA_DIR, filename))
    return new_files

def combine_datasets(existing_path, new_files):
    """Combine existing training data with new data"""
    # Load existing data
    df = pd.read_excel(existing_path)
    
    # Load and append new data
    for filepath in new_files:
        if filepath.endswith('.xlsx'):
            new_df = pd.read_excel(filepath)
        else:
            new_df = pd.read_csv(filepath)
        df = pd.concat([df, new_df], ignore_index=True)
    
    # Remove duplicates
    df.drop_duplicates(subset=['job_id'], keep='last', inplace=True)
    
    return df

def archive_files(files):
    """Move processed files to archive"""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    for filepath in files:
        filename = os.path.basename(filepath)
        new_path = os.path.join(ARCHIVE_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        os.rename(filepath, new_path)

def retrain_model():
    """Main retraining function"""
    new_files = check_for_new_data()
    
    if not new_files:
        print("No new data found. Skipping retraining.")
        return
    
    print(f"Found {len(new_files)} new data file(s). Starting retraining...")
    
    try:
        # Combine datasets
        combined_data = combine_datasets(TRAINING_DATA, new_files)
        
        # Save backup of current model
        if os.path.exists(MODEL_PATH):
            backup_path = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            os.rename(MODEL_PATH, backup_path)
        
        # Retrain model
        print("Training new model...")
        model = train_model(combined_data)
        
        # Save new training data
        combined_data.to_excel(TRAINING_DATA, index=False)
        
        # Archive processed files
        archive_files(new_files)
        
        print("Model retraining completed successfully!")
        return True
        
    except Exception as e:
        print(f"Retraining failed: {str(e)}")
        # Restore backup if exists
        if 'backup_path' in locals() and os.path.exists(backup_path):
            os.rename(backup_path, MODEL_PATH)
        return False

if __name__ == '__main__':
    retrain_model()