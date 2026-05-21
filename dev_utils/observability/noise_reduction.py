def apply_noise_reduction(log_stream):
    """Strips transient network timeouts from pure observability signals."""
    return [log for log in log_stream if "transient" not in log.get("tags", [])]
