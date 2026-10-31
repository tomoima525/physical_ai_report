"""Shared data contract; no ROS or ML dependencies."""
import math

FPS = 20
JOINTS = tuple('crane_plus_joint' + str(i) for i in (1, 2, 3, 4)) + ('crane_plus_joint_hand',)

def ordered(names, positions, wanted=JOINTS):
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError('Invalid joint message')
    mapping = dict(zip(names, positions))
    result = [float(mapping[name]) for name in wanted]
    if not all(math.isfinite(v) for v in result):
        raise ValueError('Nonfinite joint positions')
    return result

def validate_frame(frame):
    for key, size in [('observation.state', 5), ('observation.environment_state', 3), ('action', 5)]:
        value = frame[key]
        if len(value) != size or not all(math.isfinite(x) for x in value):
            raise ValueError(f'Invalid {key}')
    if frame['joint_names'] != list(JOINTS):
        raise ValueError('Joint ordering mismatch')
    if frame['object_frame'] != 'crane_plus_base':
        raise ValueError('Object frame mismatch')
    if not math.isfinite(frame['timestamp']):
        raise ValueError('Invalid timestamp')
    return frame

def held_above(heights, initial_z, fps=FPS, lift=0.04, hold=1.0):
    needed = math.ceil(fps * hold)
    return len(heights) >= needed and all(z >= initial_z + lift for z in heights[-needed:])
