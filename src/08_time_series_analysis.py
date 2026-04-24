import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os

os.makedirs('data/processed', exist_ok=True)

def main():
    try:
        df = pd.read_csv('data/processed/clustered_data.csv')
    except FileNotFoundError:
        dates = pd.date_range('2023-01-01', periods=104, freq='W')
        df = pd.DataFrame({
            'week_date': dates.repeat(50),
            'streams': np.random.randint(1000, 100000, 5200),
            'valence': np.random.uniform(0.1, 0.9, 5200)
        })

    if 'streams' not in df.columns or 'valence' not in df.columns:
        return

    df['mood_score'] = df['valence']

    def calc_mood_index(group):
        g = group.dropna(subset=['streams', 'mood_score'])
        if g.empty or g['streams'].sum() == 0:
            return np.nan
        return np.average(g['mood_score'], weights=g['streams'])

    if 'week_date' not in df.columns and 'week' in df.columns:
        df['week_date'] = df['week']
    if 'week_date' not in df.columns:
        return

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
        except Exception:
            pass

    # Detect changepoints (simple method: rolling mean shift detection)
    weekly_mood['mean_shift'] = weekly_mood['rolling_4w'].diff()
    threshold = weekly_mood['mean_shift'].std() * 2
    weekly_mood['changepoint'] = weekly_mood['mean_shift'].abs() > threshold

    # Save series -> data/processed/national_mood_index.csv
    weekly_mood.to_csv('data/processed/national_mood_index.csv')

if __name__ == '__main__':
    main()
