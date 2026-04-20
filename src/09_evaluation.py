import pandas as pd
import numpy as np
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import ttest_rel
import joblib

os.makedirs('outputs/figures', exist_ok=True)

def evaluate_clustering(report: dict):
    print("Evaluating Clustering...")
    try:
        df = pd.read_csv('data/processed/clustered_data.csv')
        model = joblib.load('outputs/models/kmeans_model.pkl')
        
        features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']
        X = df[features].dropna()
        labels = df.loc[X.index, 'cluster']
        
        # We need the scaled data for clustering metrics if the model was trained on scaled data.
        # But we'll try to calculate metrics directly on the input data for simplicity, or re-scale.
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        inertia = float(model.inertia_)
        sil_score = float(silhouette_score(X_scaled, labels))
        db_index = float(davies_bouldin_score(X_scaled, labels))
        ch_index = float(calinski_harabasz_score(X_scaled, labels))

        cluster_dist = df['cluster'].value_counts().to_dict()
        cluster_dist_named = df['mood'].value_counts().to_dict()

        report['clustering_metrics'] = {
            'silhouette_score': sil_score,
            'davies_bouldin_index': db_index,
            'calinski_harabasz_index': ch_index,
            'inertia': inertia,
            'cluster_size_distribution': cluster_dist,
            'cluster_mood_distribution': cluster_dist_named
        }
    except Exception as e:
        print(f"Error evaluating clustering: {e}")
        report['clustering_metrics'] = {"error": str(e)}

def evaluate_association_rules(report: dict):
    print("Evaluating Association Rules...")
    try:
        rules = pd.read_csv('outputs/association_rules.csv')
        
        # Top 10 rules
        top_lift = rules.sort_values('lift', ascending=False).head(10)[['antecedents', 'consequents', 'lift']].to_dict('records')
        top_confidence = rules.sort_values('confidence', ascending=False).head(10)[['antecedents', 'consequents', 'confidence']].to_dict('records')
        top_support = rules.sort_values('support', ascending=False).head(10)[['antecedents', 'consequents', 'support']].to_dict('records')

        # Coverage analysis (antecedent support or consequent support)
        coverage = {
            'mean_antecedent_support': float(rules['antecedent support'].mean()),
            'mean_consequent_support': float(rules['consequent support'].mean())
        }

        report['association_rules_metrics'] = {
            'top_10_by_lift': top_lift,
            'top_10_by_confidence': top_confidence,
            'top_10_by_support': top_support,
            'coverage_analysis': coverage
        }

        # Visualize rules as network graph using networkx
        G = nx.DiGraph()
        for _, row in rules.head(20).iterrows():
            ant = str(row['antecedents'])
            con = str(row['consequents'])
            G.add_edge(ant, con, weight=row['lift'])

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=0.5)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, edge_color='gray', font_size=10, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title('Top 20 Association Rules Graph (by Lift)')
        plt.savefig('outputs/figures/association_rules_network.png')
        plt.close()

    except Exception as e:
        print(f"Error evaluating association rules: {e}")
        report['association_rules_metrics'] = {"error": str(e)}

def evaluate_time_series(report: dict):
    print("Evaluating Time Series...")
    try:
        ts = pd.read_csv('data/processed/national_mood_index.csv')
        time_col = 'week' if 'week' in ts.columns else 'week_date'
        ts[time_col] = pd.to_datetime(ts[time_col])
        ts.set_index(time_col, inplace=True)

        events = pd.read_csv('data/raw/india_event_timeline.csv')
        events['start_date'] = pd.to_datetime(events['start_date'])
        
        # Pre/post event comparison (paired t-test)
        # Event-wise mood shift (percentage change)
        
        pre_post_results = []
        shifts = []
        severities = []
        mood_dips = []

        for _, event in events.iterrows():
            event_date = event['start_date']
            severity = event.get('severity_score', np.nan)
            
            # Simple pre/post matching (e.g., 4 weeks before, 4 weeks after)
            pre_period = ts.loc[event_date - pd.Timedelta(weeks=4):event_date - pd.Timedelta(weeks=1)]
            post_period = ts.loc[event_date + pd.Timedelta(weeks=1):event_date + pd.Timedelta(weeks=4)]

            if len(pre_period) >= 2 and len(post_period) >= 2:
                # Ensure equal length for paired t-test
                min_len = min(len(pre_period), len(post_period))
                pre_vals = pre_period['mood_index'].values[:min_len]
                post_vals = post_period['mood_index'].values[:min_len]

                t_stat, p_val = ttest_rel(pre_vals, post_vals)
                
                mean_pre = pre_period['mood_index'].mean()
                mean_post = post_period['mood_index'].mean()
                pct_change = ((mean_post - mean_pre) / mean_pre) * 100
                
                pre_post_results.append({
                    'event_name': event.get('event_name', str(event_date)),
                    't_statistic': float(t_stat),
                    'p_value': float(p_val)
                })
                
                shifts.append({
                    'event_name': event.get('event_name', str(event_date)),
                    'percent_change': float(pct_change)
                })

                if not pd.isna(severity):
                    severities.append(severity)
                    mood_dips.append(mean_post - mean_pre)

        correlation = float(np.nan)
        if severities and mood_dips:
            correlation = float(np.corrcoef(severities, mood_dips)[0, 1])

        report['time_series_metrics'] = {
            'pre_post_paired_t_test': pre_post_results,
            'event_wise_mood_shift': shifts,
            'correlation_severity_mood_dip': correlation
        }

    except Exception as e:
        print(f"Error evaluating time series: {e}")
        report['time_series_metrics'] = {"error": str(e)}

def main():
    report = {}
    
    evaluate_clustering(report)
    evaluate_association_rules(report)
    evaluate_time_series(report)
    
    with open('outputs/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("Evaluation report saved to outputs/evaluation_report.json")

if __name__ == '__main__':
    main()
