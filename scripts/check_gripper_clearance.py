#!/usr/bin/env python3
"""Read-only geometry analysis. No ROS connection or motion commands."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pickup.clearance import swept_lowest, wrist_from_joints


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/pickup.json')
    parser.add_argument('--episode',help='Optional episode directory containing frames.jsonl')
    parser.add_argument('--mesh-dir')
    parser.add_argument('--compare-offset',type=float,help='Optional grasp z offset to compare')
    parser.add_argument('--table-z-base',type=float,default=0.0,
        help='Table plane height in robot base frame; lecture setup default=0')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if args.mesh_dir:
        mesh=Path(args.mesh_dir)
    else:
        roots=[Path(__file__).resolve().parents[2]/'ros2/ai_robot_book_data/airobot_ws/src',
               Path('/home/ubuntu/airobot_ws/src')]
        candidates=[p/'crane_plus/crane_plus_description/meshes/collision' for p in roots]
        mesh=next((p for p in candidates if p.is_dir()),None)
        if mesh is None:
            parser.error('CRANE+ meshes not found; specify --mesh-dir')
    initial_z=args.table_z_base+cfg['cube_size']/2
    print('Assumptions: lecture CRANE+ collision meshes and URDF, zero endtip offset, horizontal table.')
    print('This checks hand/link4 vs table only; cube contact, arm collisions, IK and paths are NOT checked.')
    print('mesh_dir:',mesh)
    cases = [('current',cfg['grasp_offset_xyz'][2])]
    if args.compare_offset is not None:
        cases.append(('comparison',args.compare_offset))
    for name, offset in cases:
        low,q,part=swept_lowest(mesh,cfg['ik_pitch'],cfg['gripper_open'],
            cfg['gripper_closed'],initial_z+offset)
        print(f'{name}: offset_z={offset:.6f} m; min_table_clearance={(low-args.table_z_base)*1000:.3f} mm; '
              f'limiting_joint={q:.6f} rad; mesh={part}')
    if args.episode:
        frames=[json.loads(line) for line in (Path(args.episode)/'frames.jsonl').read_text().splitlines()]
        for phase in ['grasp','close','close_settle','lift','hold']:
            rows=[f for f in frames if f.get('phase')==phase]
            if not rows:
                continue
            frame=rows[-1]
            state=frame['observation.state']
            z,pitch=wrist_from_joints(state)
            low,_,part=swept_lowest(mesh,pitch,state[4],state[4],z)
            print(f'recorded {phase}: hand={state[4]:.6f} rad; pitch={pitch:.6f} rad; '
                  f'min_table_clearance={(low-args.table_z_base)*1000:.3f} mm; mesh={part}')

if __name__=='__main__':
    main()
