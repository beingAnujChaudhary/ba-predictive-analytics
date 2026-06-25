"""
Visualization Module
Creates professional, publication-quality visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, roc_curve, 
                             precision_recall_curve, auc)
from pathlib import Path


class Visualizer:
    """Professional visualization engine"""
    
    def __init__(self, output_dir='outputs/figures', style='seaborn-v0_8-whitegrid'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(style)
        sns.set_palette("colorblind")
        sns.set_context("notebook", font_scale=1.1)
        
    def plot_target_distribution(self, y, save=True, show=True):
        """Plot target variable distribution"""
        print("\n📊 Creating target distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        value_counts = y.value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0].pie(value_counts, labels=['No Booking', 'Booking'], 
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   explode=(0.05, 0), shadow=True)
        axes[0].set_title('Booking Completion Distribution', 
                         fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = axes[1].bar(['No Booking', 'Booking'], value_counts, 
                          color=colors, edgecolor='black', linewidth=1.2)
        axes[1].set_title('Booking Counts', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Customers')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / '01_target_distribution.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_correlation_heatmap(self, df, save=True, show=True):
        """Plot correlation heatmap"""
        print("\n📊 Creating correlation heatmap...")
        
        plt.figure(figsize=(14, 10))
        numerical_df = df.select_dtypes(include=[np.number])
        corr_matrix = numerical_df.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, square=True, 
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        
        if save:
            filepath = self.output_dir / '02_correlation_heatmap.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=True, show=True):
        """Plot confusion matrix"""
        print(f"\n📊 Creating confusion matrix for {model_name}...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'},
                   xticklabels=['No Booking', 'Booking'],
                   yticklabels=['No Booking', 'Booking'])
        plt.title(f'Confusion Matrix - {model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        if save:
            filepath = self.output_dir / f'03_confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curve(self, y_true, y_proba, model_name, save=True, show=True):
        """Plot ROC curve"""
        print(f"\n📊 Creating ROC curve for {model_name}...")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#2ecc71', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', alpha=0.5)
        plt.fill_between(fpr, tpr, alpha=0.2, color='#2ecc71')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save:
            filepath = self.output_dir / f'04_roc_curve_{model_name.replace(" ", "_").lower()}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_proba, model_name, save=True, show=True):
        """Plot Precision-Recall curve"""
        print(f"\n📊 Creating PR curve for {model_name}...")
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='#e74c3c', lw=2, 
                label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.fill_between(recall, precision, alpha=0.2, color='#e74c3c')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='lower left', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save:
            filepath = self.output_dir / f'05_pr_curve_{model_name.replace(" ", "_").lower()}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_importance(self, feature_names, importances, model_name, 
                               top_n=20, save=True, show=True):
        """Plot feature importance"""
        print(f"\n📊 Creating feature importance plot for {model_name}...")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', 
                   palette='viridis', edgecolor='black')
        plt.title(f'Top {top_n} Feature Importances - {model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'06_feature_importance_{model_name.replace(" ", "_").lower()}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return importance_df