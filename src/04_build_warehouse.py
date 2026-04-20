import os
import yaml
import logging
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, BigInteger, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()

# --- SCHEMA DEFINITIONS ---

class DimSong(Base):
    __tablename__ = 'dim_song'
    song_id = Column(Integer, primary_key=True)
    track_id = Column(String, unique=True, nullable=False)
    track_name = Column(String)
    valence = Column(Float)
    energy = Column(Float)
    danceability = Column(Float)
    tempo = Column(Float)
    acousticness = Column(Float)
    speechiness = Column(Float)
    loudness = Column(Float)
    duration_ms = Column(Integer)

class DimArtist(Base):
    __tablename__ = 'dim_artist'
    artist_id = Column(Integer, primary_key=True)
    artist_name = Column(String, unique=True, nullable=False)
    genre = Column(String, nullable=True)
    country = Column(String, default='India')

class DimWeek(Base):
    __tablename__ = 'dim_week'
    week_id = Column(Integer, primary_key=True)
    week_start_date = Column(Date, unique=True, nullable=False)
    week_number = Column(Integer)
    month = Column(Integer)
    quarter = Column(Integer)
    year = Column(Integer)

class DimEventPeriod(Base):
    __tablename__ = 'dim_event_period'
    event_id = Column(Integer, primary_key=True)
    event_name = Column(String, unique=True)
    start_date = Column(Date)
    end_date = Column(Date)
    event_type = Column(String)
    severity = Column(Integer)

class FactChartEntry(Base):
    __tablename__ = 'fact_chart_entry'
    chart_id = Column(Integer, primary_key=True, autoincrement=True)
    song_id = Column(Integer, ForeignKey('dim_song.song_id'))
    artist_id = Column(Integer, ForeignKey('dim_artist.artist_id'))
    week_id = Column(Integer, ForeignKey('dim_week.week_id'))
    event_id = Column(Integer, ForeignKey('dim_event_period.event_id'))
    rank = Column(Integer)
    streams = Column(BigInteger)
    mood_cluster = Column(String, nullable=True)  # Will be populated later in clustering phase
    mood_score = Column(Float)

# Indexes for query performance
Index('idx_fact_song', FactChartEntry.song_id)
Index('idx_fact_artist', FactChartEntry.artist_id)
Index('idx_fact_week', FactChartEntry.week_id)
Index('idx_fact_event', FactChartEntry.event_id)

# --- ETL FUNCTIONS ---

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found.")
        return None

def populate_dim_song(df, engine):
    logger.info("Populating dim_song...")
    song_cols = ['track_id', 'track_name', 'valence', 'energy', 'danceability', 
                 'tempo', 'acousticness', 'speechiness', 'loudness', 'duration_ms']
    
    # Filter existing columns to prevent errors if some are missing at this stage
    existing_cols = [c for c in song_cols if c in df.columns]
    songs_df = df[existing_cols].drop_duplicates(subset=['track_id']).copy()
    
    # Deduplicate and assign surrogate keys
    songs_df.to_sql('dim_song_temp', engine, if_exists='replace', index=False)
    with engine.begin() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO dim_song ({cols})
            SELECT {cols} FROM dim_song_temp
        """.format(cols=",".join(existing_cols)))
        conn.execute("DROP TABLE dim_song_temp")

def populate_dim_artist(df, engine):
    logger.info("Populating dim_artist...")
    if 'artist_name' in df.columns:
        artists_df = df[['artist_name']].drop_duplicates().copy()
        artists_df['genre'] = None
        artists_df['country'] = 'India'
        
        artists_df.to_sql('dim_artist_temp', engine, if_exists='replace', index=False)
        with engine.begin() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO dim_artist (artist_name, genre, country)
                SELECT artist_name, genre, country FROM dim_artist_temp
            """)
            conn.execute("DROP TABLE dim_artist_temp")

def populate_dim_week(df, engine):
    logger.info("Populating dim_week...")
    if 'week_date' in df.columns:
        weeks_df = df[['week_date', 'week_number', 'month', 'quarter', 'year']].drop_duplicates(subset=['week_date']).copy()
        weeks_df.rename(columns={'week_date': 'week_start_date'}, inplace=True)
        # Ensure date format as string for SQLite compatibility
        weeks_df['week_start_date'] = pd.to_datetime(weeks_df['week_start_date']).dt.date
        
        weeks_df.to_sql('dim_week_temp', engine, if_exists='replace', index=False)
        with engine.begin() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO dim_week (week_start_date, week_number, month, quarter, year)
                SELECT week_start_date, week_number, month, quarter, year FROM dim_week_temp
            """)
            conn.execute("DROP TABLE dim_week_temp")

def populate_dim_event(events_df, engine):
    logger.info("Populating dim_event_period...")
    
    # Add a "Normal" event default record if not present
    normal_event = pd.DataFrame([{
        'event_name': 'Normal',
        'start_date': None,
        'end_date': None,
        'event_type': 'Normal',
        'severity': 0
    }])
    events_df = pd.concat([normal_event, events_df], ignore_index=True)
    events_df.drop_duplicates(subset=['event_name'], inplace=True)

    events_df['start_date'] = pd.to_datetime(events_df['start_date']).dt.date
    events_df['end_date'] = pd.to_datetime(events_df['end_date']).dt.date

    events_df.to_sql('dim_event_temp', engine, if_exists='replace', index=False)
    with engine.begin() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO dim_event_period (event_name, start_date, end_date, event_type, severity)
            SELECT event_name, start_date, end_date, event_type, severity FROM dim_event_temp
        """)
        conn.execute("DROP TABLE dim_event_temp")

def populate_fact_table(df, engine):
    logger.info("Populating fact_chart_entry...")
    
    # Use pandas to query surrogate keys back
    dim_song = pd.read_sql_table('dim_song', engine)
    dim_artist = pd.read_sql_table('dim_artist', engine)
    dim_week = pd.read_sql_table('dim_week', engine)
    dim_event = pd.read_sql_table('dim_event_period', engine)
    
    dim_week['week_start_date'] = pd.to_datetime(dim_week['week_start_date'])
    df['week_date'] = pd.to_datetime(df['week_date'])
    
    # Map foreign keys
    fact_df = df.copy()
    
    # Join Songs
    fact_df = fact_df.merge(dim_song[['song_id', 'track_id']], on='track_id', how='left')
    
    # Join Artists
    fact_df = fact_df.merge(dim_artist[['artist_id', 'artist_name']], on='artist_name', how='left')
    
    # Join Weeks
    fact_df = fact_df.merge(dim_week[['week_id', 'week_start_date']], left_on='week_date', right_on='week_start_date', how='left')
    
    # Join Events
    if 'event_name' not in fact_df.columns:
        fact_df['event_name'] = 'Normal'
    fact_df = fact_df.merge(dim_event[['event_id', 'event_name']], on='event_name', how='left')
    
    # Select fact columns
    cols = ['song_id', 'artist_id', 'week_id', 'event_id', 'rank', 'streams', 'mood_score']
    if 'mood_cluster' in fact_df.columns:
        cols.append('mood_cluster')
        
    fact_ready = fact_df[[c for c in cols if c in fact_df.columns]].copy()
    
    # Insert via to_sql chunked
    fact_ready.to_sql('fact_chart_entry', engine, if_exists='append', index=False, chunksize=1000)

def main():
    config = load_config()
    if not config:
        return
        
    processed_file = os.path.join(config['paths']['processed_data'], "cleaned_merged_data.csv")
    events_file = os.path.join(config['paths']['raw_data'], "india_event_timeline.csv")
    warehouse_db = config['paths']['warehouse']
    
    if not os.path.exists(processed_file) or not os.path.exists(events_file):
        logger.error("Required data files missing. Please run preprocessing phase.")
        return
        
    df = pd.read_csv(processed_file)
    events_df = pd.read_csv(events_file)
    
    # Database Setup
    os.makedirs(os.path.dirname(warehouse_db), exist_ok=True)
    engine = create_engine(f"sqlite:///{warehouse_db}")
    
    logger.info("Initializing Data Warehouse schema...")
    Base.metadata.create_all(engine)
    
    # ETL Process
    populate_dim_song(df, engine)
    populate_dim_artist(df, engine)
    populate_dim_week(df, engine)
    populate_dim_event(events_df, engine)
    populate_fact_table(df, engine)
    
    logger.info("Data Warehouse successfully populated!")

if __name__ == "__main__":
    main()
