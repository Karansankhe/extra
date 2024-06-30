import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load and preprocess data (replace with your data loading code)
def load_and_preprocess_data():
    # Example: Load data from CSV
    df = pd.read_csv('waste_data.csv')
    
    # Preprocess data: feature engineering, cleaning, etc.
    # Example: Convert categorical variables, handle missing values, etc.
    
    return df

# Train Random Forest model for waste prediction
def train_waste_prediction_model(df):
    # Define features and target
    X = df[['feature1', 'feature2', ...]]  # Replace with relevant features
    y = df['waste_volume']  # Replace with target variable
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize Random Forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Evaluate model
def evaluate_model(model, X_test, y_test):
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')

# Example usage
if __name__ == '__main__':
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train model
    trained_model, X_test, y_test = train_waste_prediction_model(df)
    
    # Evaluate model
    evaluate_model(trained_model, X_test, y_test)
