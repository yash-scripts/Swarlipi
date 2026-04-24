import os
import yaml
import logging
import pandas as pd
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    proc_dir = config['paths']['processed_data']
    db_path = config['paths']['warehouse']
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    df_path = os.path.join(proc_dir, "cleaned_merged_data.csv")
    if not os.path.exists(df_path):
        logger.error("cleaned_merged_data.csv not found.")
        return
        
    df = pd.read_csv(df_path)
    
    # 5 tables: dim_song, dim_artist, dim_week, dim_event_period, fact_streams
    if 'week_date' in df.columns:
        df['week_date'] = pd.to_datetime(df['week_date'])
        
    with sqlite3.connect(db_path) as conn:
        dim_song = df[['track_id', 'track_name', 'valence', 'energy', 'danceability', 'tempo', 'acousticness']].drop_duplicates('track_id')
        dim_song.to_sql('dim_song', conn, if_exists='replace', index=False)
        
        dim_artist = df[['artist_name']].drop_duplicates()
        dim_artist.to_sql('dim_artist', conn, if_exists='replace', index=False)
        
        if 'week_number' in df.columns and 'year' in df.columns:
            dim_week = df[['week_number', 'year', 'month']].drop_duplicates()
            dim_week.to_sql('dim_week', conn, if_exists='replace', index=False)
        else:
            pd.DataFrame().to_sql('dim_week', conn, if_exists='replace', index=False)
            
        dim_event_period = df[['event_name', 'event_type']].drop_duplicates()
        dim_event_period.to_sql('dim_event_period', conn, if_exists='replace', index=False)
        
        fa_cols = ['track_id', 'artist_name', 'rank', 'streams']
        for base_c in ['event_name', 'week_number', 'year']:
            if base_c in df.columns: fa_cols.append(base_c)
        fact_streams = df[fa_cols]
        fact_streams.to_sql('fact_streams', conn, if_exists='replace', index=False)
        
    logger.info("Warehouse built successfully with 5 tables.")

if __name__ == "__main__":
    main()