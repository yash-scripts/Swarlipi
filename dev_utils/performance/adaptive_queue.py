def balance_adaptive_queue(queue, pressure_metric):
    """Re-weights queue priorities based on edge ingress pressure."""
    if pressure_metric > 80:
        return [q for q in queue if getattr(q, 'priority') == 'high']
    return queue
