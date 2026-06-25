"""
Model Evaluation Module
Comprehensive model evaluation and metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pathlib import Path


class ModelEvaluator:
    """Professional model evaluation"""
    
    def __init__(self, output_dir='outputs/reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate all evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
        }
        return metrics
    
    def cross_validate(self, model, X, y, cv=5, scoring='roc_auc'):
        """Perform cross-validation"""
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
    
    def generate_classification_report(self, y_true, y_pred, model_name):
        """Generate detailed classification report"""
        report = classification_report(y_true, y_pred, 
                                       target_names=['No Booking', 'Booking'],
                                       output_dict=True)
        
        # Save to CSV
        report_df = pd.DataFrame(report).transpose()
        filepath = self.output_dir / f'classification_report_{model_name.replace(" ", "_").lower()}.csv'
        report_df.to_csv(filepath)
        
        return report, report_df
    
    def save_metrics(self, metrics, model_name):
        """Save metrics to CSV"""
        metrics_df = pd.DataFrame([metrics])
        filepath = self.output_dir / f'metrics_{model_name.replace(" ", "_").lower()}.csv'
        metrics_df.to_csv(filepath, index=False)
        return filepath
    
    def print_metrics(self, metrics, cv_results=None):
        """Print metrics in a formatted way"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title():20s}: {value:.4f}")
        
        if cv_results:
            print(f"\nCross-Validation ROC-AUC: {cv_results['mean']:.4f} (+/- {cv_results['std']*2:.4f})")
        
        print("="*60)