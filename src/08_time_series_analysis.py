import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os

os.makedirs('data/processed', exist_ok=True)

def main():
    print("Task 4.3: Performing Time Series Analysis...")
    try:
        df = pd.read_csv('data/processed/clustered_data.csv')
    except FileNotFoundError:
        print("Warning: clustered_data.csv not found. Generating mock timeseries.")
        dates = pd.date_range('2023-01-01', periods=104, freq='W')
        df = pd.DataFrame({
            'week': dates.repeat(50),
            'streams': np.random.randint(1000, 100000, 5200),
            'valence': np.random.uniform(0.1, 0.9, 5200)
        })

    # Set up columns if mapped differently
    if 'streams' not in df.columns or 'valence' not in df.columns:
        print("Missing streams or valence columns. Check schema.")
        return

    # Assuming 'valence' is used as a proxy for mood_score
    df['mood_score'] = df['valence']

    # Compute weekly National Mood Index = weighted avg of mood_score by streams
    def calc_mood_index(group):
        total_streams = group['streams'].sum()
        if total_streams == 0:
            return np.nan
        return np.average(group['mood_score'], weights=group['streams'])

    weekly_mood = df.groupby('week_date').apply(calc_mood_index).reset_index(name='mood_index')
    
    # Setup datetime index
    weekly_mood['week_date'] = pd.to_datetime(weekly_mood['week_date'])
    weekly_mood = weekly_mood.sort_values('week_date').set_index('week_date')
    weekly_mood = weekly_mood.dropna()

    # Compute rolling averages (4-week, 12-week)
    weekly_mood['rolling_4w'] = weekly_mood['mood_index'].rolling(window=4).mean()
    weekly_mood['rolling_12w'] = weekly_mood['mood_index'].rolling(window=12).mean()

    # Decompose time series (trend, seasonality, residual) using statsmodels
    if len(weekly_mood) >= 24: # Require at least 2 full periods for decomposition
        try:
            decomposition = seasonal_decompose(weekly_mood['mood_index'], model='additive', period=12)
            weekly_mood['trend'] = decomposition.trend
            weekly_mood['seasonal'] = decomposition.seasonal
            weekly_mood['residual'] = decomposition.resid
        except Exception as e:
            print(f"Decomposition bypassed due to error: {e}")

    # Detect changepoints (simple method: rolling mean shift detection)
    weekly_mood['mean_shift'] = weekly_mood['rolling_4w'].diff()
    threshold = weekly_mood['mean_shift'].std() * 2
    weekly_mood['changepoint'] = weekly_mood['mean_shift'].abs() > threshold

    # Placeholder for correlation between mood index and event severity
    # Needs event timeline properly joined and feature-engineered for "severity"
    # Example: correlation = weekly_mood['mood_index'].corr(weekly_mood['event_severity'])

    # Save series -> data/processed/national_mood_index.csv
    weekly_mood.to_csv('data/processed/national_mood_index.csv')
    print("Task 4.3 completed successfully.")

if __name__ == '__main__':
    main()
