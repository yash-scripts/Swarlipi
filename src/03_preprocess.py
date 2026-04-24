import pandas as pd
import numpy as np
import yaml
import os
import logging
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found.")
        return None

def load_data(config):
    raw_dir = config['paths']['raw_data']
    
    charts_path = os.path.join(raw_dir, "spotify_india_charts.csv")
    features_path = os.path.join(raw_dir, "audio_features.csv")
    events_path = os.path.join(raw_dir, "india_event_timeline.csv")
    
    logger.info("Loading charts, features, and events data...")
    charts = pd.read_csv(charts_path)
    features = pd.read_csv(features_path)
    events = pd.read_csv(events_path)
    
    return charts, features, events

def merge_data(charts, features):
    logger.info("Merging charts and audio features on 'track_id' (inner join)...")
    return pd.merge(charts, features, on='track_id', how='inner')

def clean_data(df):
    logger.info("Cleaning data (dropping nulls, removing duplicates, capping outliers)...")
    
    # Drop rows with null audio features
    feature_cols = ['valence', 'energy', 'danceability', 'tempo', 'acousticness', 'duration_ms']
    df = df.dropna(subset=feature_cols)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Parse week_date to datetime
    if 'week_date' in df.columns:
        df['week_date'] = pd.to_datetime(df['week_date'])
    
    # Cap tempo at 99th percentile
    if 'tempo' in df.columns:
        tempo_99 = df['tempo'].quantile(0.99)
        df['tempo'] = np.where(df['tempo'] > tempo_99, tempo_99, df['tempo'])
    
    # Ensure streams is numeric, stripping commas if string
    if 'streams' in df.columns:
        if df['streams'].dtype == object:
            df['streams'] = df['streams'].astype(str).str.replace(',', '').astype(float).astype(int)
            
        Q1 = df['streams'].quantile(0.25)
        Q3 = df['streams'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        df['streams'] = np.where(df['streams'] > upper_bound, upper_bound, df['streams'])
        
    return df

def engineer_features(df):
    logger.info("Engineering new features (mood_score, date components, log_streams)...")
    
    if all(col in df.columns for col in ['valence', 'energy', 'danceability']):
        df['mood_score'] = 0.5 * df['valence'] + 0.3 * df['energy'] + 0.2 * df['danceability']
        
    if 'week_date' in df.columns:
        df['year'] = df['week_date'].dt.year
        df['month'] = df['week_date'].dt.month
        df['quarter'] = df['week_date'].dt.quarter
        df['week_number'] = df['week_date'].dt.isocalendar().week
        
    if 'streams' in df.columns:
        df['log_streams'] = np.log1p(df['streams'])
        
    return df

def tag_events(df, events):
    logger.info("Tagging events by matching 'week_date' range...")
    
    # Ensure correct types
    events['start_date'] = pd.to_datetime(events['start_date'])
    events['end_date'] = pd.to_datetime(events['end_date'])
    
    # Initialize defaults
    df['event_name'] = "Normal"
    df['event_type'] = "Normal"
    # We map severity as numeric, Normal=0
    df['event_severity'] = 0
    
    # Evaluate dates for each event
    for _, row in events.iterrows():
        mask = (df['week_date'] >= row['start_date']) & (df['week_date'] <= row['end_date'])
        df.loc[mask, 'event_name'] = row['event_name']
        df.loc[mask, 'event_type'] = row['event_type']
        df.loc[mask, 'event_severity'] = row['severity']
        
    return df

def normalize_features(df, config):
    logger.info("Applying MinMaxScaler on clustering features...")
    
    cluster_features = config['clustering']['features']
    scaler = MinMaxScaler()
    
    # We apply scaling only on columns that exist in the dataframe
    cols_to_scale = [col for col in cluster_features if col in df.columns]
    
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
    return df

def main():
    config = load_config()
    if not config:
        return
        
    try:
        charts, features, events = load_data(config)
    except FileNotFoundError as e:
        logger.error(f"Missing required data file: {e}")
        return
        
    df = merge_data(charts, features)
    df = clean_data(df)
    df = engineer_features(df)
    df = tag_events(df, events)
    df = normalize_features(df, config)
    
    # Save Output
    processed_dir = config['paths']['processed_data']
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "cleaned_merged_data.csv")
    
    df.to_csv(out_path, index=False)
    logger.info(f"Preprocessing complete. Cleaned data saved at: {out_path}")

if __name__ == "__main__":
    main()
