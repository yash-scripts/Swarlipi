import pandas as pd
import os
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load the project configuration from YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def collect_charts():
    """
    Process the global Spotify Charts dataset to filter and clean data 
    specifically for India between 2020 and 2023.
    """
    config = load_config()
    raw_data_dir = config['paths']['raw_data']
    input_path = os.path.join(raw_data_dir, "charts.csv")
    output_path = os.path.join(raw_data_dir, "spotify_india_charts.csv")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}.")
        return

    logger.info("Loading charts.csv...")
    df = pd.read_csv(input_path)
    
    logger.info("Filtering for India and date range (2020-2023)...")
    df = df[df['region'].str.lower() == 'india'].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2023-12-31')]
    
    logger.info("Extracting track_id and renaming columns...")
    df['track_id'] = df['url'].str.split('/').str[-1]
    df.rename(columns={
        'title': 'track_name',
        'date': 'week_date',
        'artist': 'artist_name'
    }, inplace=True)
    
    logger.info("Dropping duplicates and keeping lowest rank...")
    df = df.sort_values('rank')
    df = df.drop_duplicates(subset=['track_id', 'week_date'], keep='first')
    
    df.to_csv(output_path, index=False)
    
    total_rows = len(df)
    unique_tracks = df['track_id'].nunique()
    date_range = f"{df['week_date'].min().date()} to {df['week_date'].max().date()}"
    
    print(f"Total rows: {total_rows}")
    print(f"Unique tracks: {unique_tracks}")
    print(f"Date range: {date_range}")
    logger.info("Finished processing charts.")

if __name__ == "__main__":
    collect_charts()
