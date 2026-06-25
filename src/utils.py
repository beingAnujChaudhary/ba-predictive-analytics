"""
Utility Functions
Helper functions for the project
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json


def create_timestamp():
    """Create timestamp for file naming"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_dataframe_to_csv(df, filepath, description=''):
    """Save dataframe to CSV with metadata"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved {description} to {filepath}")


def save_json(data, filepath):
    """Save dictionary to JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"✓ Saved JSON to {filepath}")


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_model_metadata(model_name, metrics, cv_results, feature_names, timestamp=None):
    """Create model metadata dictionary"""
    if timestamp is None:
        timestamp = create_timestamp()
    
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'cross_validation': {
            'mean_roc_auc': float(cv_results['mean']),
            'std_roc_auc': float(cv_results['std'])
        },
        'feature_names': feature_names
    }
    return metadata


def print_section_header(text, char='='):
    """Print a formatted section header"""
    print("\n" + char*70)
    print(text)
    print(char*70)


def print_subsection_header(text, char='-'):
    """Print a formatted subsection header"""
    print("\n" + char*60)
    print(text)
    print(char*60)