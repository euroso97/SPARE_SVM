# svmtrainer_project/examples/sample_usage.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming your package is installed or the svmtrainer_project directory is in PYTHONPATH
from sparesvm import SVMModel, load_tabular_data #, load_tabular_data # Uncomment if you use it

def run_spare_ad_train():

    csv_path = '/home/kylebaik/Test_Projects/NiChart_Paper_SPARE_Experiments/lists/SPARE-AD-Harmonized.csv'

    try:
        print(f"\nAttempting to load data from {csv_path} using load_tabular_data...")
        features_df_loaded, labels_series_loaded = load_tabular_data(csv_path, 'DX', ignore_columns=['MRID', 'Sex', 'Study', 'SITE'])
        
        X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(
            features_df_loaded, 
            labels_series_loaded, 
            test_size=0.2, 
            random_state=42 #, stratify=labels_series_loaded
        )
        print("Data loaded and split successfully from CSV.")
        # You can choose to use X_train_csv, y_train_csv etc. for the rest of the script
        X_train, y_train, X_test, y_test = X_train_csv, y_train_csv, X_test_csv, y_test_csv
    
    except Exception as e:
        print(f"Could not run CSV loading example: {e}")

    # Example with a linear kernel and tuning
    print("\n--- Linear Kernel SVM without Hyperparameter Tuning ---")
    svm_trainer_linear = SVMModel(kernel='linear', tune_hyperparameters=True, random_state=42)
    print("Training Linear SVM model...")
    svm_trainer_linear.train(X_train, y_train)
    
    print("Linear SVM Details:", svm_trainer_linear.get_model_details())
    # Save the linear model
    svm_trainer_linear.save_model('/home/kylebaik/Projects/NiChart_SPARE/models/SPARE_AD/spare_ad_svm.joblib')

    # Reporting
    predictions_linear = svm_trainer_linear.predict(X_test)
    accuracy_linear = accuracy_score(y_test, predictions_linear)
    print(f"\nAccuracy of Linear SVM on test set: {accuracy_linear:.4f}")
    print(classification_report(y_test, predictions_linear, target_names=['CN','AD']))


if __name__ == "__main__":
    run_spare_ad_train()
    print("\nSPARE-AD Training finished.")