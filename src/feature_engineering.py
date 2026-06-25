"""
Feature Engineering Module
Creates new features from existing data
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Professional feature engineering"""
    
    def __init__(self):
        self.new_features = []
        
    def create_features(self, df):
        """Create new features"""
        print("🔧 Creating new features...")
        df_eng = df.copy()
        
        # 1. Route complexity
        if 'route' in df_eng.columns:
            df_eng['route_complexity'] = df_eng['route'].apply(
                lambda x: len(str(x).split('-')) if pd.notna(x) else 1
            )
            self.new_features.append('route_complexity')
            print(f"  ✓ Created route_complexity")
        
        # 2. Is long haul (flight_duration > 6 hours)
        if 'flight_duration' in df_eng.columns:
            df_eng['is_long_haul'] = (df_eng['flight_duration'] > 6).astype(int)
            self.new_features.append('is_long_haul')
            print(f"  ✓ Created is_long_haul")
        
        # 3. Lead time category
        if 'purchase_lead' in df_eng.columns:
            df_eng['lead_time_category'] = pd.cut(
                df_eng['purchase_lead'],
                bins=[0, 7, 30, 90, 180, float('inf')],
                labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            )
            self.new_features.append('lead_time_category')
            print(f"  ✓ Created lead_time_category")
        
        # 4. Stay duration category
        if 'length_of_stay' in df_eng.columns:
            df_eng['stay_category'] = pd.cut(
                df_eng['length_of_stay'],
                bins=[0, 7, 30, 90, float('inf')],
                labels=['Short', 'Medium', 'Long', 'Extended']
            )
            self.new_features.append('stay_category')
            print(f"  ✓ Created stay_category")
        
        # 5. Weekend flight
        if 'flight_day' in df_eng.columns:
            df_eng['is_weekend'] = df_eng['flight_day'].isin(['Sat', 'Sun']).astype(int)
            self.new_features.append('is_weekend')
            print(f"  ✓ Created is_weekend")
        
        # 6. Number of add-ons
        addon_cols = ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']
        available_addons = [col for col in addon_cols if col in df_eng.columns]
        if available_addons:
            df_eng['add_ons'] = sum(df_eng[col].astype(int) for col in available_addons)
            self.new_features.append('add_ons')
            print(f"  ✓ Created add_ons")
        
        # 7. Peak hour flight
        if 'flight_hour' in df_eng.columns:
            df_eng['is_peak_hour'] = df_eng['flight_hour'].apply(
                lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0
            )
            self.new_features.append('is_peak_hour')
            print(f"  ✓ Created is_peak_hour")
        
        # 8. Price sensitivity proxy
        if all(col in df_eng.columns for col in ['num_passengers', 'length_of_stay', 'purchase_lead']):
            df_eng['price_sensitivity'] = (
                df_eng['num_passengers'] * df_eng['length_of_stay']
            ) / (df_eng['purchase_lead'] + 1)
            self.new_features.append('price_sensitivity')
            print(f"  ✓ Created price_sensitivity")
        
        print(f"\n  Total new features created: {len(self.new_features)}")
        return df_eng
    
    def get_feature_list(self):
        """Return list of created features"""
        return self.new_features