import pandas as pd
import yaml
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found.")
        return None

def generate_report():
    config = load_config()
    if not config:
        return
        
    processed_file = os.path.join(config['paths']['processed_data'], "cleaned_merged_data.csv")
    
    if not os.path.exists(processed_file):
        logger.error(f"Processed data not found at {processed_file}. Please run src/03_preprocess.py first.")
        return
        
    logger.info(f"Reading {processed_file} for data quality report...")
    df = pd.read_csv(processed_file)
    
    # 1. Missing value report
    missing_report = df.isnull().sum().to_frame(name='Missing Values')
    missing_report['% Missing'] = (missing_report['Missing Values'] / len(df)) * 100
    
    # 2. Distribution statistics (describe)
    desc_report = df.describe().transpose()
    
    # Add styling and generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Data Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            h1, h2 {{ color: #1DB954; }} /* Spotify green for theme */
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 0.9em; }}
            th, td {{ border: 1px solid #ddd; text-align: right; padding: 10px; }}
            th {{ background-color: #f8f9fa; text-align: center; }}
            .text-left {{ text-align: left; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .summary {{ padding: 15px; background: #e9ecef; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body class="container">
        <h1>Data Quality Report</h1>
        
        <div class="summary">
            <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Records:</strong> {len(df)}</p>
            <p><strong>Total Features:</strong> {df.shape[1]}</p>
            <p><strong>Source File:</strong> <code>{processed_file}</code></p>
        </div>

        <h2>Missing Values Report</h2>
        {missing_report.to_html(classes=['table', 'missing-table'], float_format=lambda x: f'{{:.2f}}%'.format(x) if '%' in str(x) else x)}

        <h2>Distribution Statistics (Numeric Metrics)</h2>
        {desc_report.to_html(classes=['table', 'desc-table'], float_format=lambda x: f'{{:.4f}}'.format(x))}

        <p style="text-align:center; color:#888; font-size:0.8em; margin-top:50px;">
            Music Mood Mining India Capstone - Data Quality Analysis
        </p>
    </body>
    </html>
    """
    
    # Ensure outputs directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "data_quality_report.html")
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(html_content)
        
    logger.info(f"Data quality report successfully saved to {report_path}")

if __name__ == "__main__":
    generate_report()
