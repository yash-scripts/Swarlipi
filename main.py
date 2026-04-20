"""
Main pipeline runner.
PRN: YOUR_PRN_HERE
"""
import logging
import importlib

collect_charts = importlib.import_module("src.01_collect_charts")
fetch_audio_features = importlib.import_module("src.02_fetch_audio_features")
preprocess = importlib.import_module("src.03_preprocess")
build_warehouse = importlib.import_module("src.04_build_warehouse")
olap_queries = importlib.import_module("src.05_olap_queries")
kmeans_clustering = importlib.import_module("src.06_kmeans_clustering")
association_rules = importlib.import_module("src.07_association_rules")
time_series_analysis = importlib.import_module("src.08_time_series_analysis")
evaluation = importlib.import_module("src.09_evaluation")
visualizations = importlib.import_module("src.10_visualizations")

def run_pipeline():
    logging.info("STEP 1/10: Collecting chart data...")
    collect_charts.run()
    
    logging.info("STEP 2/10: Fetching audio features...")
    fetch_audio_features.run()
    
    logging.info("STEP 3/10: Preprocessing...")
    preprocess.run()
    
    logging.info("STEP 4/10: Building warehouse...")
    build_warehouse.run()
    
    logging.info("STEP 5/10: Running OLAP queries...")
    olap_queries.run()
    
    logging.info("STEP 6/10: K-Means clustering...")
    kmeans_clustering.run()
    
    logging.info("STEP 7/10: Association rules...")
    association_rules.run()
    
    logging.info("STEP 8/10: Time series analysis...")
    time_series_analysis.run()
    
    logging.info("STEP 9/10: Evaluation...")
    evaluation.run()
    
    logging.info("STEP 10/10: Generating visualizations...")
    visualizations.run()
    
    logging.info("✅ PIPELINE COMPLETE")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
