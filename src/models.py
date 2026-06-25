"""
Model Training Module
Train and save machine learning models
"""

import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from pathlib import Path


class ModelTrainer:
    """Professional model training"""
    
    def __init__(self, models_dir='models', random_state=42):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.trained_models = {}
        
    def train_logistic_regression(self, X_train, y_train, tune=False):
        """Train Logistic Regression"""
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*60)
        
        if tune:
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = GridSearchCV(
                LogisticRegression(random_state=self.random_state, 
                                 class_weight='balanced', max_iter=1000),
                param_grid, cv=5, scoring='roc_auc', n_jobs=-1
            )
            model.fit(X_train, y_train)
            print(f"✓ Best parameters: {model.best_params_}")
        else:
            model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state,
                C=0.1,
                solver='liblinear'
            )
            model.fit(X_train, y_train)
        
        self.trained_models['Logistic Regression'] = model
        print("✓ Logistic Regression trained successfully")
        return model
    
    def train_random_forest(self, X_train, y_train, tune=False):
        """Train Random Forest"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        if tune:
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 12, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = GridSearchCV(
                RandomForestClassifier(random_state=self.random_state,
                                     class_weight='balanced', n_jobs=-1),
                param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            model.fit(X_train, y_train)
            print(f"✓ Best parameters: {model.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        self.trained_models['Random Forest'] = model
        print("✓ Random Forest trained successfully")
        return model
    
    def save_model(self, model, model_name):
        """Save trained model"""
        filepath = self.models_dir / f'{model_name.replace(" ", "_").lower()}.pkl'
        joblib.dump(model, filepath)
        print(f"✓ Model saved to {filepath}")
        return filepath
    
    def save_all_models(self):
        """Save all trained models"""
        saved_paths = {}
        for name, model in self.trained_models.items():
            saved_paths[name] = self.save_model(model, name)
        return saved_paths