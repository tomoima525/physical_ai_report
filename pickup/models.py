"""Physical cube model shared by visual and collision geometry."""
import math
from html import escape


def cube_sdf(cfg):
    size = float(cfg['cube_size'])
    mass = float(cfg.get('cube_mass', 0.02))
    xyz = [float(v) for v in cfg['spawn_world_xyz']]
    if not math.isfinite(size) or size <= 0 or not math.isfinite(mass) or mass <= 0:
        raise ValueError('Cube size and mass must be positive and finite')
    if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
        raise ValueError('Invalid cube spawn position')
    inertia = mass*size*size/6
    position = ' '.join(str(v) for v in xyz)
    name = escape(cfg['object_name'], quote=True)
    return f'''<sdf version="1.6"><model name="{name}"><pose>{position} 0 0 0</pose>
<link name="body"><inertial><mass>{mass}</mass><inertia><ixx>{inertia}</ixx><iyy>{inertia}</iyy><izz>{inertia}</izz></inertia></inertial>
<collision name="collision"><geometry><box><size>{size} {size} {size}</size></box></geometry>
<surface><friction><ode><mu>1.5</mu><mu2>1.5</mu2></ode></friction></surface></collision>
<visual name="visual"><geometry><box><size>{size} {size} {size}</size></box></geometry><material><diffuse>0.9 0.2 0.1 1</diffuse></material></visual>
</link></model></sdf>'''
