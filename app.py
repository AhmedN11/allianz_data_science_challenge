import pandas as pd
from allianz_data_science_challenge.modeling.train import train_model
from allianz_data_science_challenge.modeling.predict import predict
from allianz_data_science_challenge.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "best_model.pkl"

def main():
    """Main function to train model if needed and make a prediction."""
    print("Bank Marketing Model - Prediction")
    print("=" * 40)
    
    # Sample input data - replace with actual feature names from your dataset
    sample_data = {
        'age': 39,
        'job': 'technician',
        'marital': 'married',
        'education': 'high.school',
        'default': 'no',
        'housing': 'yes',
        'loan': 'no',
        'contact': 'telephone',
        'month': 'may',
        'day_of_week': 'fri',
        'duration': 346,
        'campaign': 1,
        'pdays': 999,
        'previous': 0,
        'poutcome': 'nonexistent',
        'emp.var.rate': 1.1,
        'cons.price.idx': 93.994,
        'cons.conf.idx': -36.4,
        'euribor3m': 4.857,
        'nr.employed': 5191.0
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([sample_data])
    print(f"Input data: {sample_data}")
    
    # Try to make prediction
    try:
        predictions = predict(df)
        print(f"Prediction: {int(predictions[0])}")
    except Exception as e:
        # If prediction fails, train the model first
        print(f"Prediction failed: {e}")
        print("Training model...")
        
        try:
            train_model()
            print("Model trained successfully!")
            
            # Try prediction again
            predictions = predict(df)
            pred = int(predictions[0])
            if pred == 0 :
                print(f"Prediction is 0.\nThis client is predicted to not subscribe to a term deposit.")
            elif pred == 1:
                print(f"Prediction is 1.\nThis client is predicted to subscribe to a term deposit.")
        except Exception as train_error:
            print(f"Training failed: {train_error}")

if __name__ == "__main__":
    main()
