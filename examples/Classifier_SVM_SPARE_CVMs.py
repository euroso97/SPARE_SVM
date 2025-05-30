# svmtrainer_project/examples/sample_usage.py
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score

# Assuming your package is installed or the svmtrainer_project directory is in PYTHONPATH
from sparesvm import SVMTrainer, load_tabular_data #, load_tabular_data # Uncomment if you use it

def run_spare_cvm_train():

    list_cvms = ['Hypertension','Hyperlipidemia','Smoking','Obesity','Diabetes']

    for c in ['Hypertension','Hyperlipidemia','Smoking','Obesity','Diabetes']:
        csv_path = "/home/kylebaik/Test_Projects/20250528_SPARE_Validation/lists/cvm_lists_from_Sindhuja/SPARE-%s.csv" % c

        try:
            print(f"\nAttempting to load data from {csv_path} using load_tabular_data...")
            features_df_loaded, labels_series_loaded = load_tabular_data(csv_path, 
                                                                         c,
                                                                         ignore_columns=['MRID','Sex'])
            X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(
                features_df_loaded, labels_series_loaded, test_size=0.2, shuffle=True, random_state=42
            )
            print("Data loaded and split successfully from CSV.")
            # You can choose to use X_train_csv, y_train_csv etc. for the rest of the script
            X_train, y_train, X_test, y_test = X_train_csv, y_train_csv, X_test_csv, y_test_csv
        
        except Exception as e:
            print(f"Could not run CSV loading example: {e}")

        # Example with a linear kernel and no tuning
        print("\n--- Linear Kernel SVM without Hyperparameter Tuning ---")
        svm_trainer_linear = SVMTrainer(kernel='linear', tune_hyperparameters=False, random_state=42)
        print("Training Linear SVM model...")
        svm_trainer_linear.train(X_train, y_train)
        predictions_linear = svm_trainer_linear.predict(X_test)
        
        print(f"\ROC-AUC of Linear SVM on test set: {roc_auc_score(y_test, predictions_linear):.4f}")
        print(f"\nAccuracy of Linear SVM on test set: {accuracy_score(y_test, predictions_linear):.4f}")
        print(f"\nBalanced Accuracy of Linear SVM on test set: {balanced_accuracy_score(y_test, predictions_linear):.4f}")
        #print("Linear SVM Details:", svm_trainer_linear.get_model_details())
        
        print(classification_report(y_test, predictions_linear, target_names=['Control',c]))


if __name__ == "__main__":
    run_spare_cvm_train()
    print("\nSPARE-CVMs training finished.")