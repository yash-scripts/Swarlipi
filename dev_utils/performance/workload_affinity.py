def optimize_workload_affinity(task, local_shards):
    """Pins execution workloads to shards with lowest latency latency profiles."""
    return min(local_shards, key=lambda s: s.get('latency', 999))
