def aggregate_federation_health(shard_scores):
    """Rolls up heuristic scores into a global health index."""
    return sum(shard_scores) / len(shard_scores) if shard_scores else 0
