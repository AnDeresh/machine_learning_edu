from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

def split_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the raw dataframe into training and validation sets.
    
    Parameters:
    - raw_df: pd.DataFrame - The raw input dataframe.
    
    Returns:
    - Tuple containing training features, validation features, training targets, validation targets.
    """
    X = raw_df.drop('Exited', axis=1)
    y = raw_df[['Exited']]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=29)
    return X_train, X_val, y_train, y_val

def scale_numeric_features(X_train: pd.DataFrame, X_val: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric features using MinMaxScaler.
    
    Parameters:
    - X_train: pd.DataFrame - Training features.
    - X_val: pd.DataFrame - Validation features.
    - numeric_cols: List[str] - List of numeric columns to scale.
    
    Returns:
    - Tuple containing scaled training features, scaled validation features, and the scaler object.
    """
    scaler = MinMaxScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    return X_train, X_val, scaler

def encode_geography(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Encode the 'Geography' column using OneHotEncoder.
    
    Parameters:
    - X_train: pd.DataFrame - Training features.
    - X_val: pd.DataFrame - Validation features.
    
    Returns:
    - Tuple containing encoded training features, encoded validation features, the encoder object, and the list of encoded column names.
    """
    encoder = OneHotEncoder(sparse_output=False)
    encoded_cols = encoder.fit(X_train[['Geography']]).get_feature_names_out().tolist()
    X_train[encoded_cols] = encoder.transform(X_train[['Geography']])
    X_val[encoded_cols] = encoder.transform(X_val[['Geography']])
    return X_train, X_val, encoder, encoded_cols

def encode_gender(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode the 'Gender' column to numerical values.
    
    Parameters:
    - X_train: pd.DataFrame - Training features.
    - X_val: pd.DataFrame - Validation features.
    
    Returns:
    - Tuple containing encoded training features and encoded validation features.
    """
    gender_map = {'Female': 0, 'Male': 1}
    X_train['Is_Male'] = X_train['Gender'].map(gender_map)
    X_val['Is_Male'] = X_val['Gender'].map(gender_map)
    return X_train, X_val

def drop_columns(X_train: pd.DataFrame, X_val: pd.DataFrame, columns_to_drop: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop specified columns from the training and validation sets.
    
    Parameters:
    - X_train: pd.DataFrame - Training features.
    - X_val: pd.DataFrame - Validation features.
    - columns_to_drop: List[str] - List of columns to drop.
    
    Returns:
    - Tuple containing the modified training features and validation features.
    """
    X_train.drop(columns=columns_to_drop, inplace=True)
    X_val.drop(columns=columns_to_drop, inplace=True)
    return X_train, X_val

def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Dict[str, Any]:
    """
    Preprocess the data using a pipeline of functions.
    
    Parameters:
    - raw_df: pd.DataFrame - The raw input dataframe.
    - scaler_numeric: bool - Whether to scale numeric features or not.
    
    Returns:
    - Dictionary containing preprocessed data and transformation objects.
    """
    # Split data
    X_train, X_val, y_train, y_val = split_data(raw_df)
    
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    categorical_cols.remove('Surname')
    
    # Scale numeric columns if required
    scaler = None
    if scaler_numeric:
        X_train, X_val, scaler = scale_numeric_features(X_train, X_val, numeric_cols)
    
    # Drop 'Surname' column
    X_train.drop('Surname', axis=1, inplace=True)
    X_val.drop('Surname', axis=1, inplace=True)

    # Encode categorical features
    X_train, X_val, encoder, encoded_cols = encode_geography(X_train, X_val)
    X_train, X_val = encode_gender(X_train, X_val)

    # Drop unnecessary columns
    drop_cols = ['id', 'CustomerId', 'Geography', 'Gender']
    X_train, X_val = drop_columns(X_train, X_val, drop_cols)

    # Prepare the final input columns list
    input_cols = X_train.columns.tolist()

    result = {
        'X_train': X_train, 
        'y_train': y_train, 
        'X_val': X_val, 
        'y_val': y_val, 
        'input_cols': input_cols, 
        'scaler': scaler, 
        'encoder': encoder
    }
    
    return result

def preprocess_new_data(new_data: pd.DataFrame, scaler: Optional[MinMaxScaler], encoder: OneHotEncoder, input_cols: List[str]) -> Dict[str, Any]:
    """
    Preprocess new data using the provided scaler and encoder.
    
    Parameters:
    - new_data: pd.DataFrame - New data to preprocess.
    - scaler: Optional[MinMaxScaler] - Scaler object for numeric features (if any).
    - encoder: OneHotEncoder - Encoder object for categorical features.
    - input_cols: List[str] - List of columns to retain after preprocessing.
    
    Returns:
    - Dictionary containing preprocessed new data and transformation objects.
    """
    # Identify numeric and categorical columns
    numeric_cols = new_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Scale numeric columns if scaler is provided
    if scaler:
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    
    # Encode categorical features
    encoded_cols = encoder.get_feature_names_out().tolist()
    new_data[encoded_cols] = encoder.transform(new_data[['Geography']])
    
    # Encode 'Gender' column
    gender_map = {'Female': 0, 'Male': 1}
    new_data['Is_Male'] = new_data['Gender'].map(gender_map)
    
    # Drop unnecessary columns
    drop_cols = ['id', 'CustomerId', 'Geography', 'Gender', 'Surname']
    new_data.drop(columns=drop_cols, inplace=True)
    
    # Retain only the input columns used for training
    new_data = new_data[input_cols]
    
    result = {
        'X_test': new_data, 
        'input_cols': input_cols, 
        'scaler': scaler, 
        'encoder': encoder
    }
    
    return result
