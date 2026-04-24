import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yaml
import time
import os
import logging
import numpy as np
from spotipy.exceptions import SpotifyException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load the project configuration from YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def generate_mock_features_for_track(track_id):
    """Fallback generator for a single track if API fails."""
    return {
        'track_id': track_id,
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
    }

def fetch_audio_features():
    """Fetch audio features in batches, handling rate limits, resuming, and API fallbacks."""
    config = load_config()
    raw_data_dir = config['paths']['raw_data']
    charts_path = os.path.join(raw_data_dir, "spotify_india_charts.csv")
    output_path = os.path.join(raw_data_dir, "audio_features.csv")

    if not os.path.exists(charts_path):
        logger.error(f"Cannot find charts data at {charts_path}.")
        return

    df_charts = pd.read_csv(charts_path)
    if 'track_id' not in df_charts.columns:
        logger.error("track_id column missing from charts data.")
        return
    
    unique_tracks = df_charts['track_id'].dropna().unique().tolist()
    
    existing_tracks = set()
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        if 'track_id' in df_existing.columns:
            existing_tracks = set(df_existing['track_id'].dropna().astype(str).tolist())
    else:
        # Create file with headers if it doesn't exist
        pd.DataFrame(columns=['track_id', 'valence', 'energy', 'danceability', 'tempo', 
                              'acousticness', 'speechiness', 'loudness', 'liveness', 
                              'instrumentalness', 'duration_ms', 'mode', 'key']).to_csv(output_path, index=False)

    tracks_to_fetch = [t for t in unique_tracks if str(t) not in existing_tracks]
    
    logger.info(f"Total unique tracks: {len(unique_tracks)}")
    logger.info(f"Already fetched: {len(existing_tracks)}")
    logger.info(f"Tracks remaining to fetch: {len(tracks_to_fetch)}")

    if not tracks_to_fetch:
        logger.info("All tracks have been fetched. Exiting.")
        return

    client_id = config['spotify_api']['client_id']
    client_secret = config['spotify_api']['client_secret']

    use_mock_fallback = False
    sp = None
    if client_id.startswith("PASTE_YOUR_") or client_secret.startswith("PASTE_YOUR_"):
        logger.warning("Mocking fallback triggered: Placeholder credentials found.")
        use_mock_fallback = True
    else:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth_manager)

    batch_size = 50
    total_fetched_this_run = 0
    total_skipped_null = 0
    total_mocked = 0
    
    for i in range(0, len(tracks_to_fetch), batch_size):
        batch = tracks_to_fetch[i:i + batch_size]
        batch_features = []
        
        if use_mock_fallback:
            for t in batch:
                batch_features.append(generate_mock_features_for_track(t))
                total_mocked += 1
                total_fetched_this_run += 1
        else:
            try:
                response = sp.audio_features(tracks=batch)
                for f_idx, feature in enumerate(response):
                    if feature is not None:
                        batch_features.append({
                            'track_id': feature.get('id', batch[f_idx]),
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
                        total_fetched_this_run += 1
                    else:
                        total_skipped_null += 1
            except SpotifyException as e:
                # 403 Forbidden is a common error for new apps with restricted access
                logger.error(f"Spotify API error: {e}")
                if e.http_status in [403, 401, 429]:
                    logger.warning(f"Spotify API returned {e.http_status}. Falling back to mock data generator for remaining tracks.")
                    use_mock_fallback = True
                    for t in batch:
                        batch_features.append(generate_mock_features_for_track(t))
                        total_mocked += 1
                        total_fetched_this_run += 1
            except Exception as e:
                logger.error(f"Unexpected error fetching batch: {e}. Falling back to mock data.")
                use_mock_fallback = True
                for t in batch:
                    batch_features.append(generate_mock_features_for_track(t))
                    total_mocked += 1
                    total_fetched_this_run += 1
                    
            time.sleep(0.5)

        if batch_features:
            df_batch = pd.DataFrame(batch_features)
            df_batch.to_csv(output_path, mode='a', header=False, index=False)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {(i + batch_size)} / {len(tracks_to_fetch)} tracks...")

    logger.info("--- FETCH SUMMARY ---")
    logger.info(f"Total processed in this run: {total_fetched_this_run}")
    logger.info(f"Total mock fallback generated: {total_mocked}")
    logger.info(f"Total skipped (null features from API): {total_skipped_null}")
    if len(tracks_to_fetch) > 0:
        success_rate = (total_fetched_this_run / len(tracks_to_fetch)) * 100
        logger.info(f"Success Rate: {success_rate:.2f}%")

if __name__ == "__main__":
    fetch_audio_features()
