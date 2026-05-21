def predict_retention_exhaustion(current_usage, growth_rate):
    """Forecasts days until telemetry warehouse retention saturation."""
    remaining = 100.0 - current_usage
    return remaining / growth_rate if growth_rate > 0 else 999
