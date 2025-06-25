from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from allianz_data_science_challenge.dataset import load_data
from allianz_data_science_challenge.features import preprocess_bank_marketing_data, save_training_stats
from allianz_data_science_challenge.config import *
import joblib
import os
import logging
import numpy as np

def train_model():
    """
    Train the bank marketing prediction model with hyperparameter optimization.
    """
    logging.info("Starting model training...")
    
    # Load raw data
    raw_data = load_data()
    logging.info(f"Raw data loaded. Shape: {raw_data.shape}")
    
    # Save training statistics before preprocessing
    save_training_stats(raw_data)
    logging.info("Training statistics saved for prediction preprocessing")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor_info = preprocess_bank_marketing_data(raw_data)
    logging.info(f"Data preprocessing completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Save preprocessor info for prediction time
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor_info, PROCESSED_DATA_DIR / "preprocessor_info.pkl")
    logging.info("Preprocessor info saved successfully")

    # Apply SMOTE for handling class imbalance
    logging.info("Applying SMOTE for class imbalance handling...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"SMOTE applied. Resampled shape: {X_resampled.shape}")
    
    # Log class distribution after SMOTE
    unique, counts = np.unique(y_resampled, return_counts=True)
    logging.info(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")

    # Define scoring metrics
    balanced_f1_scorer = make_scorer(f1_score, average='macro')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define BaggingClassifier with DecisionTreeClassifier as base estimator
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(class_weight='balanced'),
        random_state=42
    )
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', bagging_model)
    ])
    
    # Define hyperparameter search space
    search_spaces = {
        'classifier__n_estimators': Integer(50, 200),
        'classifier__max_samples': Real(0.5, 1.0),
        'classifier__max_features': Real(0.5, 1.0),
        'classifier__estimator__max_depth': Integer(3, 20),
        'classifier__estimator__min_samples_split': Integer(2, 20),
        'classifier__estimator__min_samples_leaf': Integer(1, 10),
        'classifier__estimator__max_features': ['sqrt', 'log2', None]
    }
    
    logging.info("Starting Bayesian hyperparameter optimization...")
    
    # Bayesian optimization
    bayes_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_spaces,
        n_iter=50,  # Number of parameter settings that are sampled
        cv=cv,
        scoring=balanced_f1_scorer,
        n_jobs=-1,
        random_state=42
        )
    
    # Fit the model
    bayes_search.fit(X_resampled, y_resampled)
    
    # Get the best model
    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_
    
    logging.info(f"Best cross-validation score: {best_score:.4f}")
    logging.info(f"Best parameters: {best_params}")
    
    # Evaluate on test set
    test_predictions = best_model.predict(X_test)
    test_f1 = f1_score(y_test, test_predictions, average='macro')
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    logging.info(f"Test F1 Score (macro): {test_f1:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save the best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)
    logging.info(f"Best model saved to {model_path}")
    
    # Save model metadata
    model_metadata = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'test_f1_score': test_f1,
        'test_accuracy': test_accuracy,
        'feature_names': list(X_train.columns),
        'n_features': X_train.shape[1],
        'model_type': 'BaggingClassifier with DecisionTree',
        'preprocessing_info': preprocessor_info
    }
    
    metadata_path = MODELS_DIR / "model_metadata.pkl"
    joblib.dump(model_metadata, metadata_path)
    # logging.info(f"Model metadata saved to {metadata_path}")
    
    # Save detailed results
    results = {
        'cv_results': bayes_search.cv_results_,
        'best_estimator': best_model,
        'best_params': best_params,
        'best_score': best_score,
        'test_predictions': test_predictions,
        'test_f1_score': test_f1,
        'test_accuracy': test_accuracy
    }
    
    results_path = MODELS_DIR / "training_results.pkl"
    joblib.dump(results, results_path)
    logging.info(f"Training results saved to {results_path}")
    
    return best_model, model_metadata


def evaluate_model(model=None, model_path=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model object (optional)
        model_path: Path to saved model (optional)
    """
    if model is None:
        if model_path is None:
            model_path = MODELS_DIR / "best_model.pkl"
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
    
    # Load test data
    try:
        from allianz_data_science_challenge.features import load_processed_data
        X_train, X_test, y_train, y_test = load_processed_data()
        logging.info("Processed data loaded for evaluation")
    except:
        # If processed data not available, reprocess
        logging.info("Processed data not found, reprocessing...")
        raw_data = load_data()
        X_train, X_test, y_train, y_test, _ = preprocess_bank_marketing_data(raw_data)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    f1 = f1_score(y_test, predictions, average='macro')
    accuracy = accuracy_score(y_test, predictions)
    
    logging.info(f"Model Evaluation Results:")
    logging.info(f"F1 Score (macro): {f1:.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    
    # If model supports probability predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)
        logging.info(f"Probability predictions shape: {probabilities.shape}")
    
    return {
        'f1_score': f1,
        'accuracy': accuracy,
        'predictions': predictions
    }


if __name__ == "__main__":
    import numpy as np
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Train the model
    try:
        best_model, metadata = train_model()
        logging.info("Model training completed successfully!")
        
        # Evaluate the model
        evaluation_results = evaluate_model(best_model)
        logging.info("Model evaluation completed!")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise
