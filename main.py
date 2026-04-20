import yaml
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully.")
            return config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return None

def main():
    logger.info("Music Mood Mining initialized.")
    config = load_config()
    
    if config:
        project_name = config.get("project", {}).get("name", "Unknown Project")
        logger.info(f"Project Name: {project_name}")

if __name__ == "__main__":
    main()
