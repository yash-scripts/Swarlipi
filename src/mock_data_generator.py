import pandas as pd
import numpy as np
import os
import logging
import yaml
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def generate_mock_data():
    config = load_config()
    raw_data_dir = config['paths']['raw_data']
    os.makedirs(raw_data_dir, exist_ok=True)

    charts_path = os.path.join(raw_data_dir, "spotify_india_charts.csv")
    features_path = os.path.join(raw_data_dir, "audio_features.csv")

    np.random.seed(42)

    # 1. Generate Mock Spotify Charts
    logger.info("Generating mock spotify_india_charts.csv...")
    weeks = pd.date_range(start="2020-01-01", end="2023-12-31", freq='W')
    num_tracks_per_week = 50  # Mocking top 50 instead of 200 for speed
    
    # Create 500 unique synthetic tracks
    unique_track_ids = [f"mock_track_{str(i).zfill(4)}" for i in range(500)]
    artists = [f"Artist {i}" for i in range(1, 100)]
    
    chart_data = []
    for week in weeks:
        week_tracks = np.random.choice(unique_track_ids, size=num_tracks_per_week, replace=False)
        for rank, t_id in enumerate(week_tracks, 1):
            chart_data.append({
                'rank': rank,
                'track_id': t_id,
                'track_name': f"Song {t_id.split('_')[-1]}",
                'artist_name': np.random.choice(artists),
                'streams': int(np.random.normal(500000 / rank, 100000)),
                'week_date': week.strftime('%Y-%m-%d')
            })
            
    df_charts = pd.DataFrame(chart_data)
    # Ensure streams is positive
    df_charts['streams'] = df_charts['streams'].apply(lambda x: max(1000, x))
    df_charts.to_csv(charts_path, index=False)
    
    # 2. Generate Mock Audio Features
    logger.info("Generating mock audio_features.csv...")
    features_data = []
    for t_id in unique_track_ids:
        features_data.append({
            'track_id': t_id,
            'valence': np.random.uniform(0.1, 0.9),
            'energy': np.random.uniform(0.3, 0.9),
            'danceability': np.random.uniform(0.4, 0.9),
            'tempo': np.random.uniform(80.0, 160.0),
            'acousticness': np.random.uniform(0.01, 0.8),
            'speechiness': np.random.uniform(0.03, 0.3),
            'loudness': np.random.uniform(-10.0, -3.0),
            'liveness': np.random.uniform(0.05, 0.4),
            'instrumentalness': np.random.uniform(0.0, 0.2),
            'duration_ms': int(np.random.normal(200000, 30000)),
            'mode': np.random.choice([0, 1]),
            'key': np.random.randint(0, 12)
        })

    df_features = pd.DataFrame(features_data)
    df_features.to_csv(features_path, index=False)
    
    logger.info(f"Mock data successfully saved to '{raw_data_dir}' directory.")

if __name__ == "__main__":
    generate_mock_data()
