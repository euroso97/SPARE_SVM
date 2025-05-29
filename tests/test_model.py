# svmtrainer_project/tests/test_model.py
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sparesvm import SVMModel # Assuming svmtrainer is in PYTHONPATH or installed

class TestSVMModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate synthetic data for testing
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        cls.features_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        cls.labels_series = pd.Series(y)
        cls.features_np = X
        cls.labels_np = y

    def test_initialization(self):
        model = SVMModel(kernel='linear')
        self.assertEqual(model.kernel, 'linear')
        self.assertFalse(model.tune_hyperparameters)

        model_rbf = SVMModel(kernel='rbf', tune_hyperparameters=True, C=10)
        self.assertEqual(model_rbf.kernel, 'rbf')
        self.assertTrue(model_rbf.tune_hyperparameters)
        self.assertEqual(model_rbf.C, 10) # Will be used if tuning is off, or as part of grid

    def test_invalid_kernel(self):
        with self.assertRaises(ValueError):
            SVMModel(kernel='invalid_kernel')

    def test_train_predict_linear_no_tuning_df_input(self):
        model = SVMModel(kernel='linear', tune_hyperparameters=False, random_state=42, scale_features=True)
        model.train(self.features_df, self.labels_series)
        self.assertIsNotNone(model.model)
        predictions = model.predict(self.features_df)
        self.assertEqual(len(predictions), len(self.labels_series))
        self.assertTrue(hasattr(model, '_scaler') and model._scaler is not None)


    def test_train_predict_linear_no_tuning_np_input(self):
        model = SVMModel(kernel='linear', tune_hyperparameters=False, random_state=42, scale_features=False)
        model.train(self.features_np, self.labels_np)
        self.assertIsNotNone(model.model)
        predictions = model.predict(self.features_np)
        self.assertEqual(len(predictions), len(self.labels_np))
        self.assertTrue(model._scaler is None)


    def test_train_predict_rbf_with_tuning(self):
        # Using fewer samples for faster test tuning
        X_small, y_small = make_classification(n_samples=50, n_features=4, random_state=1)
        features_small_df = pd.DataFrame(X_small)
        labels_small_series = pd.Series(y_small)

        model = SVMModel(kernel='rbf', tune_hyperparameters=True, cv_folds=2, random_state=42) # cv_folds=2 for speed
        model.train(features_small_df, labels_small_series)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.best_params_)
        self.assertIsNotNone(model.best_cv_score_)
        predictions = model.predict(features_small_df)
        self.assertEqual(len(predictions), len(labels_small_series))

    def test_predict_before_training(self):
        model = SVMModel()
        with self.assertRaises(RuntimeError):
            model.predict(self.features_df)

    def test_data_type_errors(self):
        model = SVMModel()
        with self.assertRaises(TypeError):
            model.train("not a dataframe", self.labels_series)
        with self.assertRaises(TypeError):
            model.train(self.features_df, "not a series")

    def test_data_shape_mismatch(self):
        model = SVMModel()
        wrong_labels = pd.Series(np.random.randint(0, 2, size=self.features_df.shape[0] - 1))
        with self.assertRaises(ValueError):
            model.train(self.features_df, wrong_labels)

    def test_save_load_model(self):
        import os
        model_path = "test_svm_model.joblib"

        # Train a simple model
        original_model = SVMModel(kernel='linear', C=0.5, tune_hyperparameters=False, random_state=7, scale_features=True)
        original_model.train(self.features_df.iloc[:30], self.labels_series.iloc[:30]) # Use a subset for speed
        original_model.save_model(model_path)

        self.assertTrue(os.path.exists(model_path))

        loaded_model = SVMModel.load_model(model_path)
        self.assertIsNotNone(loaded_model)
        self.assertIsNotNone(loaded_model.model)
        self.assertEqual(loaded_model.kernel, 'linear')
        self.assertEqual(loaded_model.model.get_params()['C'], 0.5) # Check if params are loaded
        self.assertTrue(loaded_model.scale_features)
        self.assertIsNotNone(loaded_model._scaler)


        # Test prediction with loaded model
        predictions = loaded_model.predict(self.features_df.iloc[30:40])
        self.assertEqual(len(predictions), 10)

        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == '__main__':
    unittest.main()