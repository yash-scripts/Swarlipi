def mitigate_topology_drift(known_state, active_state):
    """Reconciles active topology against known steady-state baselines."""
    return {k: v for k, v in active_state.items() if k in known_state}
