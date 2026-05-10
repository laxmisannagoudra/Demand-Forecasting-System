"""
Step 4: Temporal Fusion Transformer (TFT) Model for Demand Forecasting
Fixed for proper categorical variable handling
"""

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STEP 4: TEMPORAL FUSION TRANSFORMER (TFT)")
print("=" * 70)

class DemandTFTModel:
    """
    TFT Model for Walmart demand forecasting
    """
    
    def __init__(self, max_encoder_length=52, max_prediction_length=13):
        """
        Initialize TFT model
        
        Args:
            max_encoder_length: How many weeks of history to use (52 weeks = 1 year)
            max_prediction_length: How many weeks to forecast (13 weeks = 1 quarter)
        """
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.model = None
        self.training = None
        self.validation = None
        
    def prepare_data(self, df):
        """
        Prepare data for TFT model with proper categorical encoding
        """
        print("\n[INFO] Preparing data for TFT...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Create time index (weeks from start for each store)
        df = df.sort_values(['Store', 'Date'])
        df['time_idx'] = df.groupby('Store').cumcount()
        df['group_id'] = df['Store'].astype('category').cat.codes
        
        # Convert numeric categorical variables to string
        # This is critical for TFT to recognize them as categorical
        categorical_cols = ['is_holiday', 'is_holiday_season', 'is_weekend', 'is_summer']
        
        for col in categorical_cols:
            if col in df.columns:
                # Convert to string/categorical
                df[col] = df[col].astype('category')
        
        # Add a store type category
        df['store_type'] = pd.cut(
            df.groupby('Store')['Weekly_Sales'].transform('mean'),
            bins=[0, 1000000, 2000000, float('inf')],
            labels=['Small', 'Medium', 'Large']
        )
        df['store_type'] = df['store_type'].astype('category')
        
        # Define features
        # Static features (don't change over time)
        static_categoricals = ['store_type']
        
        # Time-varying known (we know these for future)
        time_varying_known_categoricals = [
            'is_holiday', 'is_holiday_season', 'is_weekend', 'is_summer'
        ]
        
        time_varying_known_reals = [
            'time_idx', 'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'year', 'month', 'quarter', 'week_of_year'
        ]
        
        # Time-varying unknown (need to predict)
        time_varying_unknown_categoricals = []
        
        # Select top numerical features to avoid overfitting
        time_varying_unknown_reals = [
            'Weekly_Sales', 'Temperature', 'CPI', 'Unemployment'
        ]
        
        # Add only the most important lag features
        important_lags = ['sales_lag_1w', 'sales_lag_2w', 'sales_lag_3w', 'sales_lag_4w']
        for lag in important_lags:
            if lag in df.columns:
                time_varying_unknown_reals.append(lag)
        
        # Add rolling means
        important_rollings = ['sales_rolling_mean_4w', 'sales_rolling_mean_8w']
        for roll in important_rollings:
            if roll in df.columns:
                time_varying_unknown_reals.append(roll)
        
        print(f"[INFO] Features selected:")
        print(f"   Static categoricals: {static_categoricals}")
        print(f"   Time-varying known categoricals: {time_varying_known_categoricals}")
        print(f"   Time-varying known reals: {len(time_varying_known_reals)}")
        print(f"   Time-varying unknown reals: {len(time_varying_unknown_reals)}")
        
        # Filter data to rows with no missing values in selected columns
        all_reals = time_varying_known_reals + time_varying_unknown_reals
        df = df.dropna(subset=all_reals)
        
        # Only include stores with enough history
        min_history = self.max_encoder_length + self.max_prediction_length
        store_counts = df.groupby('group_id').size()
        valid_stores = store_counts[store_counts >= min_history].index
        df = df[df['group_id'].isin(valid_stores)]
        
        print(f"[INFO] After filtering: {len(df)} records, {len(valid_stores)} stores")
        
        # Create TimeSeriesDataSet
        print("[INFO] Creating TimeSeriesDataSet...")
        
        try:
            self.training = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target="Weekly_Sales",
                group_ids=["group_id"],
                min_encoder_length=self.max_encoder_length // 2,
                max_encoder_length=self.max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=self.max_prediction_length,
                static_categoricals=static_categoricals,
                time_varying_known_categoricals=time_varying_known_categoricals,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_categoricals=time_varying_unknown_categoricals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(
                    groups=["group_id"], transformation="softplus"
                ),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,
            )
            
            # Create validation dataset
            self.validation = TimeSeriesDataSet.from_dataset(
                self.training, df, predict=True, stop_randomization=True
            )
            
            print(f"[OK] Training dataset size: {len(self.training)}")
            print(f"[OK] Validation dataset size: {len(self.validation)}")
            
            return self.training, self.validation
            
        except Exception as e:
            print(f"[ERROR] Failed to create TimeSeriesDataSet: {e}")
            raise
    
    def create_model(self):
        """
        Create TFT model
        """
        print("\n[INFO] Creating TFT model...")
        
        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.001,
            hidden_size=32,  # Smaller for CPU training
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,  # 7 quantiles
            loss=QuantileLoss([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
            reduce_on_plateau_patience=4,
            use_learning_rate_scheduler=True,
        )
        
        print(f"[OK] Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model
    
    def train(self, max_epochs=10):
        """
        Train TFT model
        """
        print("\n[INFO] Training TFT model...")
        
        # Create dataloaders
        train_dataloader = self.training.to_dataloader(
            train=True, batch_size=32, num_workers=0
        )
        val_dataloader = self.validation.to_dataloader(
            train=False, batch_size=32, num_workers=0
        )
        
        # Callbacks
        early_stop = pl.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, mode='min'
        )
        
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath='models/checkpoints',
            filename='tft-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=2
        )
        
        # Trainer - use CPU if GPU not available
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='cpu',  # Force CPU for compatibility
            gradient_clip_val=0.1,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=True,
            log_every_n_steps=10,
            limit_train_batches=0.5,  # Use 50% of data for faster training
            limit_val_batches=0.5
        )
        
        # Train
        try:
            trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
            print(f"[OK] Training completed!")
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            raise
        
        return trainer
    
    def predict(self, df):
        """
        Generate predictions
        """
        print("\n[INFO] Generating predictions...")
        
        # Prepare data for prediction
        df = df.copy()
        df['time_idx'] = df.groupby('Store').cumcount()
        df['group_id'] = df['Store'].astype('category').cat.codes
        
        # Convert categoricals to string
        categorical_cols = ['is_holiday', 'is_holiday_season', 'is_weekend', 'is_summer']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        df['store_type'] = pd.cut(
            df.groupby('Store')['Weekly_Sales'].transform('mean'),
            bins=[0, 1000000, 2000000, float('inf')],
            labels=['Small', 'Medium', 'Large']
        )
        df['store_type'] = df['store_type'].astype('category')
        
        # Create prediction dataset
        try:
            pred_dataset = TimeSeriesDataSet.from_dataset(
                self.training, df, predict=True, stop_randomization=True
            )
            
            # Create dataloader
            pred_dataloader = pred_dataset.to_dataloader(
                train=False, batch_size=32, num_workers=0
            )
            
            # Predict
            predictions = self.model.predict(
                pred_dataloader,
                return_y=True,
                trainer_kwargs=dict(accelerator="cpu")
            )
            
            # Get median prediction (50th percentile)
            median_pred = predictions.output[..., 3]  # Index 3 is median
            
            return median_pred.cpu().numpy()
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None
    
    def save(self, path='models/tft_model.pth'):
        """Save model"""
        torch.save(self.model.state_dict(), path)
        print(f"[SAVED] Model saved to: {path}")
    
    def load(self, path='models/tft_model.pth'):
        """Load model"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"[LOADED] Model loaded from: {path}")

def calculate_wmape(y_true, y_pred):
    """Calculate Weighted Mean Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def main():
    """
    Main execution function
    """
    # Load feature-engineered data
    print("\n[INFO] Loading feature data...")
    df = pd.read_csv('data/processed/walmart_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter to last 2 years for faster training
    cutoff_date = df['Date'].max() - pd.DateOffset(years=2)
    df = df[df['Date'] >= cutoff_date]
    print(f"[OK] Loaded {len(df)} records (last 2 years)")
    
    # Split data
    dates = sorted(df['Date'].unique())
    train_dates = dates[:-26]  # All except last 26 weeks
    test_dates = dates[-13:]   # Last 13 weeks for testing
    
    train_data = df[df['Date'].isin(train_dates)]
    test_data = df[df['Date'].isin(test_dates)]
    
    print(f"\n[INFO] Data Split for TFT:")
    print(f"   Training: {train_data['Date'].min().date()} to {train_data['Date'].max().date()}")
    print(f"     Records: {len(train_data)}")
    print(f"   Testing: {test_data['Date'].min().date()} to {test_data['Date'].max().date()}")
    print(f"     Records: {len(test_data)}")
    
    # Initialize TFT model
    tft = DemandTFTModel(max_encoder_length=26, max_prediction_length=13)  # Reduced encoder length
    
    try:
        # Prepare data
        training, validation = tft.prepare_data(train_data)
        
        # Create model
        tft.create_model()
        
        # Train model (use fewer epochs for testing)
        print("\n[INFO] Starting training (this may take 5-10 minutes)...")
        trainer = tft.train(max_epochs=5)  # Start with 5 epochs
        
        # Save model
        tft.save('models/tft_walmart_model.pth')
        
        print("\n" + "=" * 70)
        print("STEP 4 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] TFT training failed: {e}")
        print("\n[INFO] This is expected if you don't have sufficient memory or GPU.")
        print("[INFO] The simplified TFT version will work on any machine.")
        print("\nRun: python src/tft_simple.py")

if __name__ == "__main__":
    main()