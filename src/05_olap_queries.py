import os
import yaml
import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def execute_queries():
    config = load_config()
    db_path = config['paths']['warehouse']
    
    if not os.path.exists(db_path):
        logger.error(f"Data warehouse not found.")
        return
        
    conn = sqlite3.connect(db_path)
    
    try:
        logger.info("\n" + "="*50)
        logger.info("OLAP QUERY: Roll-up (Average mood per quarter per year)")
        print(pd.read_sql_query("SELECT year, week_number, COUNT(*) FROM fact_streams GROUP BY year LIMIT 5", conn))
        
        logger.info("\n" + "="*50)
        logger.info("OLAP QUERY: Drill-down etc")
        print(pd.read_sql_query("SELECT event_name, COUNT(*) FROM fact_streams GROUP BY event_name", conn))
        
    except Exception as e:
        logger.error(f"OLAP Queries failed: {e}")

if __name__ == "__main__":
    execute_queries()

    logger.info("\n" + "="*50)
    logger.info("OLAP QUERY: Slice (All songs during COVID Wave 2)")
    logger.info("="*50)
    q3 = """
        SELECT s.track_name, a.artist_name, f.streams, f.mood_score
        FROM fact_chart_entry f
        JOIN dim_song s ON f.song_id = s.song_id
        JOIN dim_artist a ON f.artist_id = a.artist_id
        JOIN dim_event_period e ON f.event_id = e.event_id
        WHERE e.event_name = 'COVID Wave 2'
        ORDER BY f.streams DESC
        LIMIT 10;
    """
    df3 = pd.read_sql_query(q3, conn)
    print(df3)

    logger.info("\n" + "="*50)
    logger.info("OLAP QUERY: Dice (Happy + Energetic songs during IPL seasons)")
    logger.info("="*50)
    # Assuming Happy/Energetic translates to high valence and high energy 
    # (using unnormalized > 0.6 if applicable, or mood_score > 0.5)
    q4 = """
        SELECT s.track_name, a.artist_name, s.valence, s.energy, e.event_name
        FROM fact_chart_entry f
        JOIN dim_song s ON f.song_id = s.song_id
        JOIN dim_artist a ON f.artist_id = a.artist_id
        JOIN dim_event_period e ON f.event_id = e.event_id
        WHERE e.event_type = 'Sports' 
          AND s.valence > 0.6 
          AND s.energy > 0.6
        ORDER BY s.valence DESC, s.energy DESC
        LIMIT 10;
    """
    df4 = pd.read_sql_query(q4, conn)
    print(df4)

    logger.info("\n" + "="*50)
    logger.info("OLAP QUERY: Pivot (Mood x Event Type matrix)")
    logger.info("="*50)
    # Using SQL to average mood by event type, but we construct a simple pivot viewing average metrics
    q5 = """
        SELECT 
            e.event_type,
            ROUND(AVG(s.valence), 4) as avg_valence,
            ROUND(AVG(s.energy), 4) as avg_energy,
            ROUND(AVG(s.danceability), 4) as avg_danceability,
            ROUND(AVG(f.mood_score), 4) as avg_mood_score,
            COUNT(f.chart_id) as total_entries
        FROM fact_chart_entry f
        JOIN dim_song s ON f.song_id = s.song_id
        JOIN dim_event_period e ON f.event_id = e.event_id
        GROUP BY e.event_type;
    """
    df5 = pd.read_sql_query(q5, conn)
    # To truly 'pivot', we can present the DataFrame pivoted via pandas for better visualization
    pivot_df = df5.set_index('event_type').transpose()
    print(pivot_df)

    conn.close()

if __name__ == "__main__":
    execute_queries()
