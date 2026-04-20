"""Main pipeline runner."""

import logging
from pathlib import Path
import runpy

BASE_DIR = Path(__file__).resolve().parent


def _run_script(script_relative_path, *entrypoints):
    script_path = BASE_DIR / script_relative_path
    module_globals = runpy.run_path(str(script_path), run_name="__swarlipi_step__")

    for entrypoint in entrypoints:
        candidate = module_globals.get(entrypoint)
        if callable(candidate):
            candidate()
            return

    available_callables = sorted(
        name for name, value in module_globals.items() if callable(value) and not name.startswith("_")
    )
    raise AttributeError(
        f"No callable entrypoint found in {script_relative_path}. "
        f"Tried: {', '.join(entrypoints)}. "
        f"Available callables: {', '.join(available_callables) if available_callables else 'none'}. "
        "Define one of the expected entrypoints or update the _run_script call."
    )


def run_pipeline():
    logging.info("STEP 1/10: Collecting chart data...")
    _run_script("src/01_collect_charts.py", "run", "collect_charts", "main")

    logging.info("STEP 2/10: Fetching audio features...")
    _run_script("src/02_fetch_audio_features.py", "run", "fetch_audio_features", "main")

    logging.info("STEP 3/10: Preprocessing...")
    _run_script("src/03_preprocess.py", "run", "main")

    logging.info("STEP 4/10: Building warehouse...")
    _run_script("src/04_build_warehouse.py", "run", "main")

    logging.info("STEP 5/10: Running OLAP queries...")
    _run_script("src/05_olap_queries.py", "run", "execute_queries", "main")

    logging.info("STEP 6/10: K-Means clustering...")
    _run_script("src/06_kmeans_clustering.py", "run", "main")

    logging.info("STEP 7/10: Association rules...")
    _run_script("src/07_association_rules.py", "run", "main")

    logging.info("STEP 8/10: Time series analysis...")
    _run_script("src/08_time_series_analysis.py", "run", "main")

    logging.info("STEP 9/10: Evaluation...")
    _run_script("src/09_evaluation.py", "run", "main")

    logging.info("STEP 10/10: Generating visualizations...")
    _run_script("src/10_visualizations.py", "run", "main")

    logging.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
