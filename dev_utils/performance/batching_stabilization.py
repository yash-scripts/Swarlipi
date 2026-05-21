def stabilize_batch_processing(raw_batch):
    """Ensures batch sizes remain within safe operational bounds for downstream."""
    return raw_batch[:500] if len(raw_batch) > 500 else raw_batch
