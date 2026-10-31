"""Simulation-time rollout scheduling and duration-aware success checks."""
import bisect
import math

TIMING_TOLERANCE = 0.015
MAX_LATENESS = 0.100
MAX_CONSECUTIVE_MISSES = 3


def finish_inference_cycle(deadline, now, consecutive_misses, strict=False):
    """Return the schedule base for the next cycle and updated miss streak.

    A recoverable miss rebases to now, so the next target is a full period
    later; never issue a burst of catch-up commands. All times are sim seconds.
    """
    if not math.isfinite(deadline) or not math.isfinite(now):
        raise ValueError('Nonfinite simulation time')
    lateness = max(0.0, now-deadline)
    if lateness <= TIMING_TOLERANCE+1e-9:
        return deadline, 0
    streak = consecutive_misses+1
    if strict or lateness > MAX_LATENESS+1e-9 or streak >= MAX_CONSECUTIVE_MISSES:
        raise RuntimeError(f'Inference timing stop: late={lateness*1000:.1f} ms, '
                           f'consecutive_misses={streak}, strict={strict}')
    return now, streak


def final_window_start(timestamps, hold):
    """Index of the sample at/before the beginning of the final hold window."""
    if not timestamps:
        return None
    if hold <= 0 or not math.isfinite(hold):
        raise ValueError('Hold duration must be positive and finite')
    if not all(math.isfinite(t) for t in timestamps):
        raise ValueError('Nonfinite frame timestamp')
    if any(b <= a for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError('Frame timestamps must increase')
    if timestamps[-1]-timestamps[0] < hold-1e-9:
        return None
    return max(0,bisect.bisect_right(timestamps,timestamps[-1]-hold+1e-9)-1)


def held_above_timed(timestamps, heights, initial_z, lift=0.04, hold=1.0, max_gap=0.2):
    if len(timestamps) != len(heights):
        raise ValueError('Timestamp and height counts differ')
    if not all(math.isfinite(z) for z in heights) or not math.isfinite(initial_z):
        raise ValueError('Nonfinite object height')
    start = final_window_start(timestamps,hold)
    if start is None:
        return False
    if any(b-a > max_gap+1e-9 for a,b in zip(timestamps[start:],timestamps[start+1:])):
        return False
    return all(z >= initial_z+lift for z in heights[start:])


def inference_timing_summary(frames):
    timestamps = [f['timestamp'] for f in frames]
    duration = timestamps[-1]-timestamps[0] if len(frames)>1 else 0.0
    lateness = [f.get('deadline_lateness_sim_ms',0.0) for f in frames]
    latency = [f['inference_wall_ms'] for f in frames]
    return {'observed_fps':(len(frames)-1)/duration if duration>0 else None,
            'recorded_duration_sim_seconds':duration,'frame_count':len(frames),
            'deadline_misses':sum(v > (TIMING_TOLERANCE+1e-9)*1000 for v in lateness),
            'max_deadline_lateness_sim_ms':max(lateness,default=0.0),
            'inference_wall_ms_mean':sum(latency)/len(latency) if latency else None,
            'inference_wall_ms_max':max(latency,default=None)}
