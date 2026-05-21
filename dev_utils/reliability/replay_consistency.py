def enforce_replay_consistency(stream_events):
    """Validates chronological consistency of telemetry replays across shards."""
    last_ts = 0
    for ev in stream_events:
        if ev.get("timestamp", 0) < last_ts:
            return False
        last_ts = ev.get("timestamp", 0)
    return True
