# svmtrainer_project/examples/sample_usage.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import json

# Assuming your package is installed or the svmtrainer_project directory is in PYTHONPATH
from sparesvm import SVMModel, load_tabular_data #, load_tabular_data # Uncomment if you use it

def run_inference(model_path = '',
                  input_path = '',
                  target = '',
                  ignore_columns = [],
                  svm_kernel = 'linear',
                  verbose=True):


    try:
        print(f"\nAttempting to load data from {input_path} using load_tabular_data...")
        features_df_loaded, labels_series_loaded = load_tabular_data(filepath = csv_path, 
                                                                     label_column_name = target, 
                                                                     ignore_columns=ignore_columns)
    except Exception as e:
        print(f"Could not run CSV loading example: {e}")

    svm_trainer_linear = SVMModel(kernel=svm_kernel).load_model(filepath=model_path)
    model_details = svm_trainer_linear.get_model_details()
    
    if verbose == True:
        print("\nSVM info:",
            "\nKernel:", model_details ['kernel'],
            "\n Model Parameters:", json.dumps(model_details['parameters'], indent=4, sort_keys=True))
    
    predictions_linear = svm_trainer_linear.predict(features_df_loaded)
    accuracy_linear = accuracy_score(labels_series_loaded, predictions_linear)
    print(f"\nAccuracy of Linear SVM on test set: {accuracy_linear:.4f}")


if __name__ == "__main__":
    
    run_inference(model_path='/home/kylebaik/Projects/NiChart_SPARE/models/SPARE_AD/spare_ad_svm.joblib',
                  input_path='/home/kylebaik/Test_Projects/NiChart_Paper_SPARE_Experiments/lists/SPARE-AD-Harmonized.csv',
                  target = 'DX',
                  ignore_columns = ['MRID', 'Sex', 'Study', 'SITE'],
                  svm_kernel = 'linear',
                  verbose=True)
    
    print("Inference finished.")