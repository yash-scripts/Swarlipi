def calculate_stability_heuristic(node_metrics):
    """Predicts node stability based on jitter over a 5-minute window."""
    jitter = node_metrics.get("jitter_ms", 0)
    return max(0, 100 - (jitter * 1.5))
