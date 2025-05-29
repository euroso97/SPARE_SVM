# svmtrainer_project/examples/sample_usage.py
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming your package is installed or the svmtrainer_project directory is in PYTHONPATH
from sparesvm import SVMModel, load_tabular_data #, load_tabular_data # Uncomment if you use it

def run_example():
    # # --- Option 1: Generate sample data ---
    # print("Generating sample classification data...")
    # X_data, y_data = make_classification(n_samples=2000, n_features=15, n_informative=5,
    #                                      n_redundant=2, n_classes=2, random_state=42)
    # features_df = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(X_data.shape[1])])
    # labels_series = pd.Series(y_data, name='target')

    # # Split data
    # X_train, X_test, y_train, y_test = train_test_split(
    #     features_df, labels_series, test_size=0.2, random_state=42, stratify=labels_series
    # )
    # print(f"Training data shape: Features {X_train.shape}, Labels {y_train.shape}")
    # print(f"Test data shape: Features {X_test.shape}, Labels {y_test.shape}")

    # --- (Optional) Option 2: Load data from a CSV ---
    # For this to work, create a 'sample_data.csv' file first.
    # print("\nCreating a dummy CSV for demonstration...")
    # sample_data_for_csv = pd.concat([features_df, labels_series], axis=1)
    # csv_path = 'sample_data.csv'
    # sample_data_for_csv.to_csv(csv_path, index=False)
    #
    list_cvms = ['Hypertension','Hyperlipidemia','Smoking','Obesity','Diabetes']

    for c in ['Hypertension','Hyperlipidemia','Smoking','Obesity','Diabetes']:
        csv_path = "/home/kylebaik/Test_Projects/20250528_SPARE_Validation/lists/cvm_lists_from_Sindhuja/SPARE-%s.csv" % c

        try:
            print(f"\nAttempting to load data from {csv_path} using load_tabular_data...")
            features_df_loaded, labels_series_loaded = load_tabular_data(csv_path, c, ignore_columns=[i for i in list_cvms if i != c] + ['InModelAs','Study','SITE','Race'])
            X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(
                features_df_loaded, labels_series_loaded, test_size=0.2, random_state=42
            )
            print("Data loaded and split successfully from CSV.")
            # You can choose to use X_train_csv, y_train_csv etc. for the rest of the script
            X_train, y_train, X_test, y_test = X_train_csv, y_train_csv, X_test_csv, y_test_csv
        except Exception as e:
            print(f"Could not run CSV loading example: {e}")


        # # Initialize the SVMModel with hyperparameter tuning for RBF kernel
        # print("\n--- RBF Kernel SVM with Hyperparameter Tuning ---")
        # svm_trainer_rbf = SVMModel(kernel='linear', tune_hyperparameters=True, cv_folds=5, random_state=42)

        # # Train the model
        # print("Training RBF SVM model...")
        # svm_trainer_rbf.train(X_train, y_train)

        # # Get model details
        # print("\nModel Details (RBF):")
        # details_rbf = svm_trainer_rbf.get_model_details()
        # for key, value in details_rbf.items():
        #     print(f"  {key}: {value}")

        # # Make predictions
        # predictions_rbf = svm_trainer_rbf.predict(X_test)
        # # print(f"\nPredictions on test set (RBF first 10): {predictions_rbf[:10]}")

        # # Evaluate the model
        # accuracy_rbf = accuracy_score(y_test, predictions_rbf)
        # print(f"\nAccuracy on test set (RBF): {accuracy_rbf:.4f}")
        # print("\nClassification Report on test set (RBF):")
        # print(classification_report(y_test, predictions_rbf))

        # # Save and load the RBF model
        # model_path_rbf = "trained_svm_rbf_model.joblib"
        # svm_trainer_rbf.save_model(model_path_rbf)
        # loaded_svm_trainer_rbf = SVMModel.load_model(model_path_rbf)
        # if loaded_svm_trainer_rbf:
        #     loaded_predictions_rbf = loaded_svm_trainer_rbf.predict(X_test)
        #     loaded_accuracy_rbf = accuracy_score(y_test, loaded_predictions_rbf)
        #     print(f"\nAccuracy of loaded RBF model on test set: {loaded_accuracy_rbf:.4f}")
        #     # print("Details of loaded RBF model:", loaded_svm_trainer_rbf.get_model_details())


        # Example with a linear kernel and no tuning
        print("\n--- Linear Kernel SVM without Hyperparameter Tuning ---")
        svm_trainer_linear = SVMModel(kernel='linear', tune_hyperparameters=False, random_state=42)
        print("Training Linear SVM model...")
        svm_trainer_linear.train(X_train, y_train)
        predictions_linear = svm_trainer_linear.predict(X_test)
        accuracy_linear = accuracy_score(y_test, predictions_linear)
        print(f"\nAccuracy of Linear SVM on test set: {accuracy_linear:.4f}")
        print("Linear SVM Details:", svm_trainer_linear.get_model_details())

        # # Example with a polynomial kernel and tuning
        # # Note: Polynomial kernel tuning can be slow, especially with a large grid or many folds.
        # # Using cv_folds=3 for a quicker example run.
        # print("\n--- Polynomial Kernel SVM with Hyperparameter Tuning ---")
        # svm_trainer_poly = SVMModel(kernel='poly', tune_hyperparameters=True, cv_folds=3, random_state=42)
        # print("Training Polynomial SVM model...")
        # svm_trainer_poly.train(X_train, y_train)
        # predictions_poly = svm_trainer_poly.predict(X_test)
        # accuracy_poly = accuracy_score(y_test, predictions_poly)
        # print(f"\nAccuracy of Polynomial SVM on test set: {accuracy_poly:.4f}")
        # print("Polynomial SVM Details:", svm_trainer_poly.get_model_details())
        # # Save and load the Poly model
        # model_path_poly = "trained_svm_poly_model.joblib"
        # svm_trainer_poly.save_model(model_path_poly)


if __name__ == "__main__":
    run_example()
    print("\nSample usage script finished.")