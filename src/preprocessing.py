"""
Data Preprocessing Module
Handles data cleaning, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path


class DataPreprocessor:
    """Professional data preprocessing pipeline"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        
    def clean_data(self, df):
        """Clean the dataset"""
        print("🔧 Cleaning data...")
        df_clean = df.copy()
        
        # Handle missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Fill numerical with median
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(
            df_clean[numerical_cols].median()
        )
        
        # Fill categorical with mode
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        print(f"  ✓ Cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        print("🔧 Encoding categorical variables...")
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"  ✓ Encoded {col}: {len(le.classes_)} categories")
        
        return df_encoded
    
    def prepare_features(self, df, target_col='booking_complete'):
        """Separate features and target"""
        print("🔧 Preparing features...")
        
        # Separate features and target
        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col] if target_col in df.columns else None
        
        # Identify column types
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  ✓ Features: {X.shape[1]} columns")
        print(f"  ✓ Numerical: {len(self.numerical_cols)}")
        print(f"  ✓ Categorical: {len(self.categorical_cols)}")
        
        return X, y
    
    def scale_features(self, X_train, X_test, numerical_cols=None):
        """Scale numerical features"""
        print("🔧 Scaling numerical features...")
        
        if numerical_cols is None:
            numerical_cols = self.numerical_cols
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Fit scaler on training data only
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(
            X_train[numerical_cols]
        )
        X_test_scaled[numerical_cols] = self.scaler.transform(
            X_test[numerical_cols]
        )
        
        print(f"  ✓ Scaled {len(numerical_cols)} numerical columns")
        return X_train_scaled, X_test_scaled
    
    def apply_smote(self, X_train, y_train, random_state=None):
        """Apply SMOTE to training data only"""
        print("🔧 Applying SMOTE to balance classes...")
        
        if random_state is None:
            random_state = self.random_state
        
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"  ✓ Before SMOTE: {len(y_train)} samples")
        print(f"  ✓ After SMOTE: {len(y_resampled)} samples")
        print(f"  ✓ Class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        
        return X_resampled, y_resampled
    
    def save_preprocessing_objects(self, save_dir='models'):
        """Save scaler and encoders"""
        print("💾 Saving preprocessing objects...")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, save_path / 'scaler.pkl')
        print(f"  ✓ Saved scaler to {save_path / 'scaler.pkl'}")
        
        # Save label encoders
        joblib.dump(self.label_encoders, save_path / 'label_encoders.pkl')
        print(f"  ✓ Saved label encoders to {save_path / 'label_encoders.pkl'}")
        
        return save_path