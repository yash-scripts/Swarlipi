import pandas as pd
import os
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def collect_charts():
    """
    Placeholder/Helper to collect Spotify regional charts.
    Normally, you would use an API or download a Kaggle dataset 
    (e.g. `kaggle datasets download -d <dataset_name>`).
    """
    config = load_config()
    raw_data_dir = config['paths']['raw_data']
    output_path = os.path.join(raw_data_dir, "spotify_india_charts.csv")
    
    if os.path.exists(output_path):
        logger.info(f"Charts data already exists at {output_path}")
        return
        
    logger.warning(
        f"Chart data not found at {output_path}.\n"
        "Please download the 'Spotify Top Charts' dataset from Kaggle or Spotify Charts,\n"
        "rename/format it to include ['rank', 'track_id', 'track_name', 'artist_name', 'streams', 'week_date'],\n"
        "and save it there. Alternatively, run src/mock_data_generator.py to generate synthetic data."
    )

if __name__ == "__main__":
    collect_charts()
