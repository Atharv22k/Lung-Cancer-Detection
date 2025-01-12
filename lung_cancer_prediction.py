import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
def load_dataset(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Preprocessing
    # Encode categorical variables
    le = LabelEncoder()
    df['GENDER'] = le.fit_transform(df['GENDER'])
    
    # Convert Yes/No to binary for target variable
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
    
    return df

# Prepare the data
def prepare_data(df):
    # Separate features and target
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return y_pred

# Main execution
def main():
    # File path for the dataset (update this with your actual file path)
    file_path = 'lung_cancer_survey.csv'
    
    # Load and preprocess the dataset
    df = load_dataset(file_path)
    
    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the model
    joblib.dump(model, 'lung_cancer_model.joblib')

if __name__ == '__main__':
    main()
