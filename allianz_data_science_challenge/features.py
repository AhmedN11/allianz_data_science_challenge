import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from allianz_data_science_challenge.config import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import logging
import joblib

def preprocess_bank_marketing_data(df, target_column='y', test_size=0.2, random_state=42):
    """
    Preprocess the bank marketing dataset including feature engineering, encoding, and selection.
    """
    data = df.copy()
    print("Starting preprocessing...")
    print(f"Original dataset shape: {data.shape}")

    # 1. Replace 'unknown' with NaN
    data = data.replace('unknown', np.nan)

    # Fill specific known columns with mode
    for col in ['education', 'job', 'marital', 'housing', 'default']:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Fill 'loan' with mean if numerical
    if 'loan' in data.columns:
        data['loan'] = pd.to_numeric(data['loan'], errors='coerce')
        data['loan'].fillna(data['loan'].mean(), inplace=True)

    # Binary conversion
    data.replace({"no": 0, "yes": 1}, inplace=True)

    # 2. Create engineered features
    data['recent_contact'] = (data['pdays'] != 999).astype(int)
    data['many_contacts'] = (data['previous'] > 2).astype(int)
    data['long_campaign'] = (data['campaign'] > data['campaign'].median()).astype(int)
    data['contacted_in_may'] = (data['month'] == 'may').astype(int)
    data['is_student_or_retired'] = data['job'].isin(['student', 'retired']).astype(int)
    data['was_success_before'] = (data['poutcome'] == 'success').astype(int)

    engineered_features = [
        'recent_contact', 'many_contacts', 'long_campaign',
        'contacted_in_may', 'is_student_or_retired', 'was_success_before'
    ]

    # Encode target
    if data[target_column].dtype == 'object':
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        target_classes = le.classes_
    else:
        target_classes = None

    # Split
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data[target_column])
    global_mean = train[target_column].mean()

    # Mean encoding
    categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    numerical_variables = [
        'age', 'duration', 'campaign', 'pdays', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
        'euribor3m', 'nr.employed'
    ]

    for col in categorical_cols:
        mean_map = train.groupby(col)[target_column].mean().to_dict()
        train[col + '_mean_enc'] = train[col].map(mean_map)
        test[col + '_mean_enc'] = test[col].map(mean_map).fillna(global_mean)

    encoded_features = [col + '_mean_enc' for col in categorical_cols]

    # Subset final data
    train = train[encoded_features + numerical_variables + engineered_features + [target_column]]
    test = test[encoded_features + numerical_variables + engineered_features + [target_column]]

    # Drop 'duration' (as it leaks info)
    train = train.drop(columns=['duration'])
    test = test.drop(columns=['duration'])

    # Feature selection using RFECV
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    rfecv = RFECV(estimator=RandomForestClassifier(random_state=random_state),
                  min_features_to_select=10, scoring='accuracy')
    rfecv.fit(X_train, y_train)

    selected_features = X_train.columns[rfecv.support_]
    logging.info(f"Selected features are {len(selected_features)}: {', '.join(selected_features)}")

    # Final selection
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    preprocessor_info = {
        'engineered_features': engineered_features,
        'encoded_features': encoded_features,
        'selected_features': list(selected_features),
        'label_encoder': target_classes,
        'categorical_columns': categorical_cols,
        'numerical_columns': numerical_variables
    }

    return X_train, X_test, y_train, y_test, preprocessor_info


def preprocess_for_prediction(df, preprocessor_info_path=None):
    """
    Preprocess new data for prediction using the same transformations as training.
    
    Args:
        df: Input DataFrame to preprocess
        preprocessor_info_path: Path to saved preprocessor info (if None, loads from default location)
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    data = df.copy()
    logging.info(f"Starting prediction preprocessing for data shape: {data.shape}")
    
    # Load preprocessor info if path provided, otherwise use default
    if preprocessor_info_path is None:
        preprocessor_info_path = PROCESSED_DATA_DIR / "preprocessor_info.pkl"
    
    try:
        preprocessor_info = joblib.load(preprocessor_info_path)
        logging.info("Loaded preprocessor info successfully")
    except FileNotFoundError:
        logging.error(f"Preprocessor info not found at {preprocessor_info_path}")
        raise FileNotFoundError("Preprocessor info file not found. Please train the model first.")
    
    # Extract info from preprocessor
    categorical_cols = preprocessor_info['categorical_columns']
    numerical_variables = preprocessor_info['numerical_columns']
    selected_features = preprocessor_info['selected_features']
    
    # 1. Replace 'unknown' with NaN
    data = data.replace('unknown', np.nan)
    
    # Load training statistics for consistent preprocessing
    try:
        training_stats = joblib.load(PROCESSED_DATA_DIR / "training_stats.pkl")
    except FileNotFoundError:
        logging.error("Training statistics not found. Please ensure model training saves these stats.")
        raise FileNotFoundError("Training statistics file not found.")
    
    # Fill missing values using training statistics
    for col in ['education', 'job', 'marital', 'housing', 'default']:
        if col in data.columns:
            mode_value = training_stats.get(f'{col}_mode', data[col].mode()[0] if not data[col].mode().empty else 'unknown')
            data[col] = data[col].fillna(mode_value)
    
    # Fill 'loan' with training mean if numerical
    if 'loan' in data.columns:
        data['loan'] = pd.to_numeric(data['loan'], errors='coerce')
        loan_mean = training_stats.get('loan_mean', data['loan'].mean())
        data['loan'].fillna(loan_mean, inplace=True)
    
    # Binary conversion
    data.replace({"no": 0, "yes": 1}, inplace=True)
    
    # 2. Create engineered features using training statistics
    data['recent_contact'] = (data['pdays'] != 999).astype(int)
    data['many_contacts'] = (data['previous'] > 2).astype(int)
    
    # Use training median for campaign threshold
    campaign_median = training_stats.get('campaign_median', data['campaign'].median())
    data['long_campaign'] = (data['campaign'] > campaign_median).astype(int)
    
    data['contacted_in_may'] = (data['month'] == 'may').astype(int)
    data['is_student_or_retired'] = data['job'].isin(['student', 'retired']).astype(int)
    data['was_success_before'] = (data['poutcome'] == 'success').astype(int)
    
    # 3. Apply mean encoding using training mappings
    mean_encodings = training_stats.get('mean_encodings', {})
    global_mean = training_stats.get('global_mean', 0.5)
    
    for col in categorical_cols:
        if col in data.columns:
            mean_map = mean_encodings.get(col, {})
            data[col + '_mean_enc'] = data[col].map(mean_map).fillna(global_mean)
    
    # 4. Select only the features that were selected during training
    encoded_features = [col + '_mean_enc' for col in categorical_cols if col in data.columns]
    available_numerical = [col for col in numerical_variables if col in data.columns and col != 'duration']
    engineered_features = [
        'recent_contact', 'many_contacts', 'long_campaign',
        'contacted_in_may', 'is_student_or_retired', 'was_success_before'
    ]
    
    # Create feature set (excluding duration as it was dropped during training)
    all_features = encoded_features + available_numerical + engineered_features
    available_features = [col for col in all_features if col in data.columns]
    
    # Subset to available features
    data = data[available_features]
    
    # Select only the features that were selected during training
    final_features = [col for col in selected_features if col in data.columns]
    missing_features = [col for col in selected_features if col not in data.columns]
    
    if missing_features:
        logging.warning(f"Missing features for prediction: {missing_features}")
        # Create missing features with default values
        for feature in missing_features:
            data[feature] = 0
    
    # Final feature selection
    data = data[selected_features]
    
    logging.info(f"Prediction preprocessing completed. Final shape: {data.shape}")
    return data


def save_processed_data(X_train, X_test, y_train, y_test):
    """
    Save processed data to the processed data directory.
    
    Args:
        X_train, X_test, y_train, y_test: Processed data splits
    """
    logging.info("Saving processed data...")
    
    # Create processed data directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    train_data = X_train.copy()
    train_data['target'] = y_train
    train_data.to_csv(PROCESSED_DATA_DIR / "train_processed.csv", index=False)
    
    # Save test data
    test_data = X_test.copy()
    test_data['target'] = y_test
    test_data.to_csv(PROCESSED_DATA_DIR / "test_processed.csv", index=False)
    
    # Save feature names
    feature_names = list(X_train.columns)
    pd.DataFrame({'features': feature_names}).to_csv(
        PROCESSED_DATA_DIR / "feature_names.csv", index=False
    )
    
    logging.info("Processed data saved successfully")


def load_processed_data():
    """
    Load previously processed data.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logging.info("Loading processed data...")
    
    train_data = pd.read_csv(PROCESSED_DATA_DIR / "train_processed.csv")
    test_data = pd.read_csv(PROCESSED_DATA_DIR / "test_processed.csv")
    
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    logging.info("Processed data loaded successfully")
    
    return X_train, X_test, y_train, y_test


def save_training_stats(df, target_column='y'):
    """
    Save training statistics needed for consistent preprocessing during prediction.
    This should be called during training to save the statistics.
    
    Args:
        df: Training dataframe
        target_column: Name of target column
    """
    logging.info("Saving training statistics...")
    
    # Create processed data directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    training_stats = {}
    
    # Save mode values for categorical columns
    for col in ['education', 'job', 'marital', 'housing', 'default']:
        if col in df.columns:
            mode_val = df[col].mode()
            training_stats[f'{col}_mode'] = mode_val[0] if not mode_val.empty else 'unknown'
    
    # Save mean for loan column
    if 'loan' in df.columns:
        loan_numeric = pd.to_numeric(df['loan'], errors='coerce')
        training_stats['loan_mean'] = loan_numeric.mean()
    
    # Save campaign median
    if 'campaign' in df.columns:
        training_stats['campaign_median'] = df['campaign'].median()
    
    # Save global mean for target
    if target_column in df.columns:
        # Convert target to numeric if needed
        target_data = df[target_column].copy()
        if target_data.dtype == 'object':
            target_data = target_data.replace({"no": 0, "yes": 1})
        training_stats['global_mean'] = target_data.mean()
    
    # Save mean encodings
    categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    mean_encodings = {}
    
    # Prepare data for mean encoding calculation
    data_for_encoding = df.copy()
    data_for_encoding = data_for_encoding.replace('unknown', np.nan)
    data_for_encoding.replace({"no": 0, "yes": 1}, inplace=True)
    
    if target_column in data_for_encoding.columns:
        target_numeric = data_for_encoding[target_column]
        if target_numeric.dtype == 'object':
            le = LabelEncoder()
            # Convert to pandas Series instead of numpy array
            target_encoded = le.fit_transform(target_numeric)
            target_numeric = pd.Series(target_encoded, index=target_numeric.index, name=target_column)
        
        for col in categorical_cols:
            if col in data_for_encoding.columns:
                try:
                    # Create a temporary dataframe to ensure proper alignment
                    temp_df = pd.DataFrame({
                        'grouping_col': data_for_encoding[col],
                        'target_col': target_numeric
                    }).dropna()  # Remove any NaN values that might cause issues
                    
                    mean_map = temp_df.groupby('grouping_col')['target_col'].mean().to_dict()
                    mean_encodings[col] = mean_map
                except Exception as e:
                    logging.warning(f"Could not create mean encoding for column {col}: {e}")
                    mean_encodings[col] = {}
    
    training_stats['mean_encodings'] = mean_encodings
    
    # Save the statistics
    joblib.dump(training_stats, PROCESSED_DATA_DIR / "training_stats.pkl")
    logging.info("Training statistics saved successfully")
