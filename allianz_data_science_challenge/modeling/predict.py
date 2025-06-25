from pathlib import Path
import pandas as pd
import joblib

import logging
import typer

from allianz_data_science_challenge.config import MODELS_DIR, PROCESSED_DATA_DIR
from allianz_data_science_challenge.features import preprocess_for_prediction

app = typer.Typer()


def predict(input_data, model_path: Path = MODELS_DIR / "best_model.pkl", preprocess=True):
    """
    Make predictions using the saved model.
    
    Args:
        input_data: Input features (pandas DataFrame or array-like)
        model_path: Path to the saved model
        preprocess: Whether to apply preprocessing to input_data
        
    Returns:
        Predictions array
    """
    # Load the trained model
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
    
    # Preprocess input data if needed
    if preprocess:
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame for preprocessing")
        
        logging.info("Preprocessing input data for prediction...")
        try:
            processed_data = preprocess_for_prediction(input_data)
            logging.info(f"Data preprocessed successfully. Shape: {processed_data.shape}")
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise
    else:
        processed_data = input_data
    
    # Make predictions
    try:
        predictions = model.predict(processed_data)
        logging.info(f"Predictions made successfully. Shape: {predictions.shape}")
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        raise


def predict_proba(input_data, model_path: Path = MODELS_DIR / "best_model.pkl", preprocess=True):
    """
    Make probability predictions using the saved model.
    
    Args:
        input_data: Input features (pandas DataFrame or array-like)
        model_path: Path to the saved model
        preprocess: Whether to apply preprocessing to input_data
        
    Returns:
        Prediction probabilities array
    """
    # Load the trained model
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
    
    # Preprocess input data if needed
    if preprocess:
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame for preprocessing")
        
        logging.info("Preprocessing input data for prediction...")
        try:
            processed_data = preprocess_for_prediction(input_data)
            logging.info(f"Data preprocessed successfully. Shape: {processed_data.shape}")
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise
    else:
        processed_data = input_data
    
    # Make probability predictions
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
            logging.info(f"Probability predictions made successfully. Shape: {probabilities.shape}")
            return probabilities
        else:
            logging.warning("Model does not support probability predictions. Using regular predictions.")
            return predict(processed_data, model_path, preprocess=False)
    except Exception as e:
        logging.error(f"Error making probability predictions: {e}")
        raise


@app.command()
def predict_from_file(
    input_file: Path = typer.Argument(..., help="Path to input CSV file"),
    output_file: Path = typer.Option(None, help="Path to save predictions (optional)"),
    model_path: Path = typer.Option(MODELS_DIR / "best_model.pkl", help="Path to trained model"),
    include_probabilities: bool = typer.Option(False, help="Include prediction probabilities")
):
    """
    Make predictions from a CSV file using the command line.
    """
    try:
        # Load input data
        logging.info(f"Loading data from {input_file}")
        input_data = pd.read_csv(input_file)
        logging.info(f"Data loaded successfully. Shape: {input_data.shape}")
        
        # Make predictions
        predictions = predict(input_data, model_path, preprocess=True)
        
        # Create results dataframe
        results = pd.DataFrame({'predictions': predictions})
        
        # Add probabilities if requested
        if include_probabilities:
            probabilities = predict_proba(input_data, model_path, preprocess=True)
            if probabilities.ndim > 1:
                for i in range(probabilities.shape[1]):
                    results[f'probability_class_{i}'] = probabilities[:, i]
            else:
                results['probability'] = probabilities
        
        # Save or display results
        if output_file:
            results.to_csv(output_file, index=False)
            logging.info(f"Predictions saved to {output_file}")
        else:
            print(results)
            
    except Exception as e:
        logging.error(f"Error in prediction pipeline: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
