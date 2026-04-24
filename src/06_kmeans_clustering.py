import pandas as pd
import numpy as np
import yaml
import os
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def map_moods(df, cluster_col='cluster'):
    # A simple semantic mapping
    centroids = df.groupby(cluster_col)[['valence', 'energy']].mean()
    mood_map = {}
    for c, row in centroids.iterrows():
        v, e = row['valence'], row['energy']
        if v > 0.5 and e > 0.5: mood_map[c] = 'Happy'
        elif v > 0.5 and e <= 0.5: mood_map[c] = 'Chill'
        elif v <= 0.5 and e > 0.5: mood_map[c] = 'Energetic'
        else: mood_map[c] = 'Melancholic'
    return df[cluster_col].map(mood_map)

def main():
    config = load_config()
    proc_dir = config['paths']['processed_data']
    df_path = os.path.join(proc_dir, "cleaned_merged_data.csv")
    out_path = os.path.join(proc_dir, "clustered_data.csv")
    db_path = config['paths']['warehouse']
    
    if not os.path.exists(df_path):
        logger.error(f"{df_path} not found.")
        return
        
    df = pd.read_csv(df_path)
    features = config['clustering']['features']
    n_clu = config['clustering']['n_clusters']
    r_state = config['clustering']['random_state']
    
    logger.info("Task 4.1: Starting KMeans Clustering...")
    valid_feats = [f for f in features if f in df.columns]
    X = df[valid_feats].fillna(0)
    
    kmeans = KMeans(n_clusters=n_clu, random_state=r_state, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    try:
        s_score = silhouette_score(X, df['cluster'])
        logger.info(f"Silhouette Score: {s_score}")
    except:
        pass
        
    df['mood'] = map_moods(df, 'cluster')
    df.to_csv(out_path, index=False)
    
    # Generate national mood index needed downstream
    if 'week_date' in df.columns:
        national = df.groupby('week_date')['valence'].mean().reset_index()
        national.rename(columns={'valence': 'mood_index'}, inplace=True)
        national.to_csv(os.path.join(proc_dir, "national_mood_index.csv"), index=False)
        
    # Update warehouse
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("ALTER TABLE fact_streams ADD COLUMN mood TEXT")
            except: pass
            
            # Simple approach: save new dataframe to fact_streams replacing old
            fact_df = df[['track_id', 'artist_name', 'rank', 'streams', 'mood']]
            for base_c in ['event_name', 'week_number', 'year']:
                if base_c in df.columns: fact_df[base_c] = df[base_c]
            fact_df.to_sql('fact_streams', conn, if_exists='replace', index=False)
    except Exception as e:
        logger.error(f"Warehouse update failed: {e}")
        
    logger.info("Task 4.1 completed successfully.")

if __name__ == "__main__":
    main()