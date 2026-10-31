"""Lowest collision-mesh vertex over the continuous hand-joint interval.

For the lecture CRANE+ URDF: hand origin=(0,-.0145,.045), axis=(-1,0,0),
hand mesh Ry(-pi/2), link4 mesh Rx(pi/2), and zero endtip offset.
Only a horizontal table and rigid hand geometry are considered.
"""
import math
import struct
from pathlib import Path


def vertices(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84 + 50*count:
        raise ValueError('Expected binary STL: '+str(path))
    return set(struct.unpack_from('<fff', raw, 96+50*i+12*j)
               for i in range(count) for j in range(3))


def minimum_trig(a, b, c, lo, hi):
    """Exact minimum of a+b*sin(q)+c*cos(q) on [lo,hi]."""
    if lo > hi:
        lo, hi = hi, lo
    angles = [lo, hi]
    critical = math.atan2(b, c)
    for k in range(math.floor((lo-critical)/math.pi)-1,
                   math.ceil((hi-critical)/math.pi)+2):
        q = critical+k*math.pi
        if lo <= q <= hi:
            angles.append(q)
    return min((a+b*math.sin(q)+c*math.cos(q), q) for q in angles)


def swept_lowest(mesh_dir, pitch, q_open, q_closed, wrist_z):
    """Return min z relative to robot base, joint angle and limiting mesh.

    Yaw does not affect height. For a fixed rigid orientation, triangle minima
    occur at vertices; angular extrema are evaluated analytically.
    """
    sp, cp = math.sin(pitch), math.cos(pitch)
    candidates = []
    for x,y,z in vertices(Path(mesh_dir)/'Link4_collision_20201013.stl'):
        candidates.append((wrist_z-sp*x+cp*y, q_open, 'link4'))
    for x,y,z in vertices(Path(mesh_dir)/'Hand_collision_20201013.stl'):
        height, q = minimum_trig(wrist_z+sp*z+cp*.045,
                                 -cp*y, cp*x, q_open, q_closed)
        candidates.append((height, q, 'hand'))
    return min(candidates)


def wrist_from_joints(joints):
    """FK height/pitch for the lecture's zero-offset link4 endtip."""
    _,q2,q3,q4,_ = joints
    z = .0714 + .083*math.cos(q2) + .0935*math.cos(q2-q3)
    return z, q2-q3-q4
