import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Page config
st.set_page_config(page_title="Swarlipi Dashboard", layout="wide", page_icon="🎵")
st.title("🎵 Swarlipi: Music Mood Mining (India)")
st.markdown("A data mining project linking Spotify India top charts with national events.")

# Define paths
DATA_DIR = "data/processed"
RAW_DIR = "data/raw"
FIGURES_DIR = "outputs/figures"

@st.cache_data
def load_data():
    try:
        clustered_df = pd.read_csv(os.path.join(DATA_DIR, "clustered_data.csv"))
        clustered_df['week_date'] = pd.to_datetime(clustered_df['week_date'])
        
        events_df = pd.read_csv(os.path.join(RAW_DIR, "india_event_timeline.csv"))
        events_df['start_date'] = pd.to_datetime(events_df['start_date'])
        events_df['end_date'] = pd.to_datetime(events_df['end_date'])
        
        national_mood = pd.read_csv(os.path.join(DATA_DIR, "national_mood_index.csv"))
        national_mood['week_date'] = pd.to_datetime(national_mood['week_date'])
        return clustered_df, events_df, national_mood
    except Exception as e:
        st.error(f"Error loading data. Ensure pipeline tasks have been successfully run. Details: {e}")
        return None, None, None

clustered_df, events_df, national_mood = load_data()

if clustered_df is not None:
    # --- Top KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    total_tracks = len(clustered_df)
    unique_songs = clustered_df['track_id'].nunique()
    avg_valence = round(clustered_df['valence'].mean(), 2)
    avg_energy = round(clustered_df['energy'].mean(), 2)

    col1.metric("Total Chart Entries", f"{total_tracks:,}")
    col2.metric("Unique Tracks", f"{unique_songs:,}")
    col3.metric("Average Valence (Positivity)", avg_valence)
    col4.metric("Average Energy", avg_energy)
    
    st.markdown("---")

    # --- National Mood Timeline ---
    st.subheader("📈 National Mood Index over Time")
    
    # Enhanced Timeline Plot
    fig_timeline = go.Figure()
    
    # Raw Weekly Mood
    fig_timeline.add_trace(go.Scatter(
        x=national_mood['week_date'], 
        y=national_mood['mood_index'], 
        mode='lines', 
        name='Weekly Positivity',
        line=dict(color='rgba(176, 196, 222, 0.5)', width=1.5) # Lighter slate gray for subtle background tracking
    ))
    
    # 4-Week Rolling Average
    if 'rolling_4w' in national_mood.columns:
        fig_timeline.add_trace(go.Scatter(
            x=national_mood['week_date'], 
            y=national_mood['rolling_4w'], 
            mode='lines', 
            name='4-Week Trend',
            line=dict(color='#00FFFF', width=3) # Bright Cyan to pop against dark mode
        ))

    # Add event highlights
    colors = {'Pandemic': 'rgba(255, 69, 0, 0.25)', 'Festival': 'rgba(50, 205, 50, 0.25)', 'Sports': 'rgba(30, 144, 255, 0.25)'}
    
    for _, row in events_df.iterrows():
        fig_timeline.add_vrect(
            x0=row['start_date'], x1=row['end_date'],
            fillcolor=colors.get(row['event_type'], 'rgba(128, 128, 128, 0.2)'),
            opacity=0.4,
            layer="below", line_width=0,
            annotation_text=f"<b>{row['event_name']}</b>", 
            annotation_position="top left",
            annotation_font=dict(size=11, color="#F0F8FF"), # White/light blue for dark mode readability
            annotation_textangle=-90
        )
        
    fig_timeline.update_layout(
        title='National Positivity Index vs Events (Interactive)',
        template="plotly_dark", # Switch to dark template for native look
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date", 
        yaxis_title="National Mood Index (Average Valence)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Add optional range slider to x-axis
    fig_timeline.update_xaxes(rangeslider_visible=True)
    
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("---")

    # --- Layout for static figures & metrics ---
    st.subheader("📊 Cluster Analysis & Association Rules")
    
    row1_c1, row1_c2 = st.columns(2)
    
    with row1_c1:
        st.markdown("**Mood Distribution**")
        mood_counts = clustered_df['mood'].value_counts().reset_index()
        mood_counts.columns = ['Mood', 'Count']
        
        fig_pie = px.pie(mood_counts, names='Mood', values='Count', color='Mood', 
                         color_discrete_map={
                             'Happy': '#FFD700', 
                             'Energetic': '#FF4500', # Deeper vibrant orange to pop on dark mode
                             'Chill': '#40E0D0', # High contrast vibrant cyan/turquoise instead of baby blue
                             'Melancholic': '#9370DB', # Brighter purple tone instead of standard royal blue
                             'Calm': '#40E0D0' # Catch-all for Calm
                         },
                         hole=0.5)
                         
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            insidetextfont=dict(color='white', size=14, family="Arial Black"), # Force legible white text inside slices
            marker=dict(line=dict(color='#262730', width=3)) # Thicker border matching dark bg
        )
        fig_pie.update_layout(showlegend=False, 
                              margin=dict(t=10, b=10, l=10, r=10),
                              height=350,
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1_c2:
        st.markdown("**Cluster Centroid Radar**")
        
        try:
            # Interactive Radar Graph via Plotly
            radar_features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']
            centroids = clustered_df.groupby('mood')[radar_features].mean().reset_index()
            
            fig_radar = go.Figure()
            
            radar_colors = {
                'Happy': '#FFD700',      # Bright yellow-gold
                'Energetic': '#FF4500',  # Bright deep orange
                'Chill': '#40E0D0',      # Bright cyan 
                'Melancholic': '#9370DB',# High-contrast purple
                'Calm': '#40E0D0'        # Sync Calm with Chill
            }
            
            for _, row in centroids.iterrows():
                mood = row['mood']
                color = radar_colors.get(mood, '#00CED1')
                fig_radar.add_trace(go.Scatterpolar(
                    r=row[radar_features].values.tolist() + [row[radar_features].values[0]],
                    theta=[f.capitalize() for f in radar_features] + [radar_features[0].capitalize()],
                    fill='toself',
                    fillcolor=color,
                    opacity=0.6, # Make fill much more solid to prevent being lost
                    line=dict(color=color, width=3), # Thicker strong line
                    name=mood
                ))
            
            fig_radar.update_layout(
                template="plotly_dark", # Force dark mode radar explicitly
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                polar=dict(
                    bgcolor='rgba(0,0,0,0)', # Ensure underlying plot matches background instead of white disk
                    radialaxis=dict(
                        visible=True, 
                        showticklabels=False, 
                        range=[0, 1], 
                        gridcolor='#555555', # Distinct but subtle grid rings
                        linecolor='#777777'
                    ),
                    angularaxis=dict(
                        gridcolor='#555555',
                        linecolor='#777777',
                        tickfont=dict(color='white', size=14)
                    )
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1,
                    font=dict(color='white', size=12)
                ),
                margin=dict(t=30, b=10, l=30, r=30),
                height=350
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        except Exception as e:
            # Fallback to static image if feature computation fails
            radar_path = os.path.join(FIGURES_DIR, "cluster_radar.png")
            if os.path.exists(radar_path):
                st.image(radar_path, use_container_width=True)
            else:
                st.warning("Cluster radar chart not found.")
            
    st.markdown("---")
    
    row2_c1, row2_c2 = st.columns(2)
    
    with row2_c1:
        st.markdown("**Association Rules Network**")
        network_path = os.path.join(FIGURES_DIR, "rules_network.png")
        if os.path.exists(network_path):
            st.image(network_path, use_container_width=True, caption="Relationships between Events and Song Moods")
        else:
            st.warning("Association rules network chart not found.")
            
    with row2_c2:
        st.markdown("**Top Tracks Explorer**")
        selected_mood = st.selectbox("Filter tracks by mood:", options=clustered_df['mood'].unique())
        
        top_songs = clustered_df[clustered_df['mood'] == selected_mood][
            ['track_name', 'artist_name', 'streams', 'valence', 'energy']
        ].sort_values('streams', ascending=False).drop_duplicates('track_name').head(10)
        
        # Enhanced dataframe view
        st.dataframe(
            top_songs, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "track_name": "Track Label",
                "artist_name": "Artist",
                "streams": st.column_config.NumberColumn(
                    "Total Streams",
                    help="Total streams on Spotify India",
                    format="%,d",
                ),
                "valence": st.column_config.ProgressColumn(
                    "Positivity (Valence)",
                    help="Musical positiveness derived from Spotify API",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "energy": st.column_config.ProgressColumn(
                    "Intensity (Energy)",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                )
            }
        )
