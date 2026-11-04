"""Inference-only configuration of ACT action chunk execution."""


def configure_action_steps(policy, requested=None):
    saved = policy.config.n_action_steps
    effective = saved if requested is None else requested
    if type(effective) is not int or not 1 <= effective <= policy.config.chunk_size:
        raise ValueError('n_action_steps must be between 1 and model chunk_size')
    if requested is not None and policy.config.temporal_ensemble_coeff is not None:
        raise ValueError('Action queue override requires temporal ensembling disabled')
    policy.config.n_action_steps = effective
    policy.reset()
    return {'saved_n_action_steps': saved, 'n_action_steps': effective,
            'chunk_size': policy.config.chunk_size}


def environment_with_elapsed(xyz, elapsed):
    import math
    if len(xyz) != 3 or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Expected xyz and finite nonnegative elapsed simulation seconds')
    return [*xyz, elapsed]


def refresh_observations(node):
    """Service ROS callbacks before checking freshness; never sends commands."""
    node.wait(0.005)
    try:
        node.ready()
        return False
    except RuntimeError as exc:
        if not str(exc).startswith(('Missing/stale joint states','Missing/stale object pose')):
            raise
    # A clock callback may have satisfied the previous deadline before queued
    # joint callbacks ran. Give reception a bounded chance, then enforce the
    # SAME freshness checks. A missing stream must still stop execution.
    node.wait(0.05)
    node.ready()
    return True
