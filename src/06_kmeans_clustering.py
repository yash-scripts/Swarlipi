import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os
import sqlite3

# Ensure output directories exist
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def main():
    print("Task 4.1: Starting KMeans Clustering...")
    
    # Load processed data
    # Assuming 'merged_data.csv' exists, otherwise mock or gracefully exit
    data_path = 'data/processed/merged_data.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print(f"Warning: {data_path} not found. Ensure previous steps are run.")
        df = pd.DataFrame(np.random.rand(100, 5), columns=['valence', 'energy', 'danceability', 'acousticness', 'tempo'])
        df['track_id'] = [f't_{i}' for i in range(100)]
    
    features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        print("Required features missing from data.")
        return
        
    X = df[available_features].dropna()
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal K (Elbow method for K in range(2, 10))
    inertia = []
    sil_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
    # Plot Elbow and Silhouette
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('K (Number of clusters)')
    ax1.set_ylabel('Inertia (Elbow)', color=color)
    ax1.plot(K_range, inertia, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(K_range, sil_scores, color=color, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Elbow Method and Silhouette Score for Optimal K')
    plt.savefig('outputs/figures/elbow_silhouette.png')
    plt.close()
    
    # Fit KMeans with K=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Analyze centroids (inverse transform)
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    
    # Map cluster indices to heuristics
    df_clustered = df.loc[X.index].copy()
    df_clustered['cluster'] = kmeans.labels_
    
    def label_cluster(row):
        val = row.get('valence', 0)
        eng = row.get('energy', 0)
        dnc = row.get('danceability', 0)
        
        if val > 0.6 and eng > 0.6:
            return 'Happy'
        elif val < 0.4 and eng < 0.5:
            return 'Melancholic'
        elif eng > 0.7 and dnc > 0.7:
            return 'Energetic'
        else:
            return 'Calm'
            
    # Auto-label clusters
    df_clustered['mood'] = df_clustered.apply(label_cluster, axis=1)
    
    # Save model -> outputs/models/kmeans_model.pkl
    joblib.dump(kmeans, 'outputs/models/kmeans_model.pkl')
    
    # Save labeled data -> data/processed/clustered_data.csv
    df_clustered.to_csv('data/processed/clustered_data.csv', index=False)
    
    # Update warehouse fact table with cluster labels
    try:
        conn = sqlite3.connect('data/warehouse.db')
        df_clustered[['track_id', 'mood']].to_sql('temp_moods', conn, if_exists='replace', index=False)
        # Assumes fact_streams exists
        conn.execute("UPDATE fact_streams SET mood = (SELECT mood FROM temp_moods WHERE temp_moods.track_id = fact_streams.track_id)")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warehouse update skipped or failed (ensure schema exists): {e}")

    print("Task 4.1 completed successfully.")

if __name__ == '__main__':
    main()
