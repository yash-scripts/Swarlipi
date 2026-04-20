import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yaml
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def fetch_audio_features():
    config = load_config()
    raw_data_dir = config['paths']['raw_data']
    charts_path = os.path.join(raw_data_dir, "spotify_india_charts.csv")
    output_path = os.path.join(raw_data_dir, "audio_features.csv")

    if not os.path.exists(charts_path):
        logger.error(f"Cannot find charts data at {charts_path}. Please run data collection or mock generator first.")
        return

    # Load charts and get unique track IDs
    df_charts = pd.read_csv(charts_path)
    if 'track_id' not in df_charts.columns:
        logger.error("track_id column missing from charts data.")
        return
    
    unique_tracks = df_charts['track_id'].dropna().unique().tolist()
    logger.info(f"Found {len(unique_tracks)} unique tracks to process.")

    # Authenticate with Spotipy
    client_id = config['spotify_api']['client_id']
    client_secret = config['spotify_api']['client_secret']

    if client_id == "YOUR_SPOTIFY_CLIENT_ID" or client_secret == "YOUR_SPOTIFY_CLIENT_SECRET":
        logger.error("Spotify API credentials not set in config.yaml.")
        return

    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    features_list = []
    batch_size = 50

    # Batch processing with rate limiting
    for i in range(0, len(unique_tracks), batch_size):
        batch = unique_tracks[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(unique_tracks) + batch_size - 1) // batch_size}")
        
        try:
            # Skip mock ids or ids that don't match typical spotify lengths if needed
            # Valid exact length of a spotify ID is typically 22
            # Here we just pass them to the API
            response = sp.audio_features(tracks=batch)
            for feature in response:
                if feature:
                    features_list.append({
                        'track_id': feature.get('id'),
                        'valence': feature.get('valence'),
                        'energy': feature.get('energy'),
                        'danceability': feature.get('danceability'),
                        'tempo': feature.get('tempo'),
                        'acousticness': feature.get('acousticness'),
                        'speechiness': feature.get('speechiness'),
                        'loudness': feature.get('loudness'),
                        'liveness': feature.get('liveness'),
                        'instrumentalness': feature.get('instrumentalness'),
                        'duration_ms': feature.get('duration_ms'),
                        'mode': feature.get('mode'),
                        'key': feature.get('key')
                    })
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
        
        # Rate limiting: 1 sec delay every 50 calls
        time.sleep(1)

    df_features = pd.DataFrame(features_list)
    df_features.to_csv(output_path, index=False)
    logger.info(f"Saved audio features to {output_path}")

if __name__ == "__main__":
    fetch_audio_features()
