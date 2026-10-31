#!/usr/bin/env python3
"""Gazebo Sim bridge, read-only diagnostics, IK teacher, collection and ACT rollout."""
import argparse
import json
import math
import random
import subprocess
import sys
import time
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pickup.core import FPS, JOINTS, held_above, ordered, validate_frame
from pickup.models import cube_sdf
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK


class Robot(Node):
    def __init__(self, config):
        super().__init__('pickup_experiment')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.cfg = config
        self.state = self.xyz = None
        self.state_at = self.pose_at = 0
        self.arm = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.hand = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.ik = self.create_client(GetPositionIK, '/compute_ik')

    def on_joints(self, msg):
        try:
            self.state = ordered(msg.name, msg.position)
            self.state_at = time.monotonic()
        except (ValueError, KeyError):
            self.state = None

    def read_poses(self, pipe):
        for line in pipe:
            try:
                world_xyz = [float(v) for v in line.split()]
                if len(world_xyz) != 3 or not all(math.isfinite(v) for v in world_xyz):
                    continue
                self.xyz = [a-b for a,b in zip(world_xyz,self.cfg['base_world_xyz'])]
                self.pose_at = time.monotonic()
            except ValueError:
                continue

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=min(0.01, max(0, end-time.monotonic())))

    def sim_now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def wait_sim_until(self, deadline):
        wall_start = time.monotonic()
        while self.sim_now() < deadline:
            if time.monotonic()-wall_start > 15:
                raise RuntimeError('Simulation clock stopped or too slow')
            rclpy.spin_once(self, timeout_sec=0.005)

    def ready(self):
        now = time.monotonic()
        if self.state is None or now - self.state_at > 0.3:
            raise RuntimeError('Missing/stale joint states')
        if self.xyz is None or now - self.pose_at > 0.3:
            raise RuntimeError('Missing/stale object pose; spawn cube and start pose bridge')

    def publish(self, action):
        if len(action) != 5 or any(not math.isfinite(x) or not lo <= x <= hi
            for x, (lo, hi) in zip(action, self.cfg['joint_limits'])):
            raise ValueError('Action violates configured joint limits')
        for pub, names, values in [(self.arm, JOINTS[:4], action[:4]), (self.hand, JOINTS[4:], action[4:])]:
            msg = JointTrajectory()
            msg.joint_names = list(names)
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in values]
            pt.time_from_start.nanosec = 100000000
            msg.points = [pt]
            pub.publish(msg)

    def solve(self, xyz, seed, gripper):
        if not self.ik.wait_for_service(timeout_sec=3):
            raise RuntimeError('MoveIt /compute_ik unavailable')
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = 'arm'
        ik.ik_link_name = 'crane_plus_link_endtip'
        ik.robot_state.joint_state.name = list(JOINTS)
        ik.robot_state.joint_state.position = seed
        ik.pose_stamped.header.frame_id = 'crane_plus_base'
        p = ik.pose_stamped.pose
        p.position.x, p.position.y, p.position.z = xyz
        yaw = math.atan2(xyz[1], xyz[0])
        pitch = self.cfg['ik_pitch']
        p.orientation.x = -math.sin(yaw/2)*math.sin(pitch/2)
        p.orientation.y = math.cos(yaw/2)*math.sin(pitch/2)
        p.orientation.z = math.sin(yaw/2)*math.cos(pitch/2)
        p.orientation.w = math.cos(yaw/2)*math.cos(pitch/2)
        ik.timeout.sec = 2
        ik.avoid_collisions = True
        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=4)
        response = future.result() if future.done() else None
        if response is None or response.error_code.val != 1:
            raise RuntimeError(f'IK failed for {xyz}; calibrate grasp offset, pitch and workspace')
        js = response.solution.joint_state
        return ordered(js.name, js.position, JOINTS[:4]) + [gripper]

    def phase(self, target, seconds, stream=None, label="move", ramp=True):
        self.ready()
        start = list(self.state)
        frames = []
        deadline = self.sim_now()
        minimum_steps = round(seconds*FPS)
        extra_steps = round(self.cfg.get('arm_settle_timeout', 5.0)*FPS)
        tolerance = self.cfg.get('arm_arrival_tolerance', 0.04)
        for i in range(minimum_steps + extra_steps):
            self.ready()
            alpha = min(1.0, (i+1)/(seconds*FPS)) if ramp else 1.0
            alpha = alpha*alpha*(3-2*alpha)
            action = [a+(b-a)*alpha for a,b in zip(start,target)]
            timestamp = self.sim_now()
            frame = validate_frame({'timestamp': timestamp, 'joint_names': list(JOINTS),
                'object_frame': 'crane_plus_base', 'phase': label, 'observation.state': list(self.state),
                'observation.environment_state': list(self.xyz), 'action': action})
            # Snapshot observation before issuing the corresponding command.
            self.publish(action)
            if stream:
                stream.write(json.dumps(frame)+'\n')
            frames.append(frame)
            deadline += 1/FPS
            if self.sim_now() > deadline + 0.015:
                raise RuntimeError('Missed 20 Hz deadline; episode invalid')
            self.wait_sim_until(deadline)
            self.ready()
            errors = [abs(a-b) for a,b in zip(self.state[:4],target[:4])]
            # Gripper contact can prevent reaching its target; do not demand full closure.
            if i+1 >= minimum_steps and max(errors) <= tolerance:
                print(f'{label}: arm arrived; hand={self.state[4]:.3f}, target={target[4]:.3f}', flush=True)
                return frames
        raise RuntimeError(f'{label}: arm did not arrive; errors_rad={errors}, '
                           f'hand={self.state[4]:.3f}, target_hand={target[4]:.3f}')


def ign_service(world, service, request_type, request):
    result = subprocess.run(['ign','service','-s',f'/world/{world}/{service}',
        '--reqtype',request_type,'--reptype','ignition.msgs.Boolean',
        '--timeout','5000','--req',request], capture_output=True, text=True, timeout=8, check=True)
    if 'data: true' not in result.stdout:
        raise RuntimeError(f'Gazebo {service} failed: {result.stdout} {result.stderr}')


def spawn(cfg):
    sdf = cube_sdf(cfg)
    ign_service(cfg['world'], 'create', 'ignition.msgs.EntityFactory',
        'sdf: '+json.dumps(sdf)+' allow_renaming: false')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['diagnose','spawn','respawn','reset','plan','scan','trial','collect','infer'])
    parser.add_argument('--config', default='configs/pickup_clearance_candidate.json',
        help='Configuration file (default: working 4 cm cube setup)')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='data/episodes')
    parser.add_argument('--model-path')
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--wait-for-recording', action='store_true',
        help='After home/object reset, wait for Enter before starting the episode')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.mode in ('spawn', 'respawn'):
        print(f'config={args.config}; cube_size={cfg["cube_size"]} m; mass={cfg.get("cube_mass", 0.02)} kg', flush=True)
        if args.mode == 'respawn':
            print(f'Replacing Gazebo model {cfg["object_name"]} with cube_size={cfg["cube_size"]} m', flush=True)
            ign_service(cfg['world'], 'remove', 'ignition.msgs.Entity',
                'name: '+json.dumps(cfg['object_name'])+' type: MODEL')
            time.sleep(0.5)
        spawn(cfg)
        return
    if args.mode == 'reset':
        x,y,z = cfg['spawn_world_xyz']
        ign_service(cfg['world'], 'set_pose', 'ignition.msgs.Pose',
            f'name: "{cfg["object_name"]}" position {{x: {x} y: {y} z: {z}}} orientation {{w: 1}}')
        return
    if args.wait_for_recording and args.episodes != 1:
        parser.error('--wait-for-recording requires --episodes 1')
    if args.mode == 'trial' and args.episodes != 1:
        parser.error('trial is limited to one calibration episode')
    rclpy.init()
    node = Robot(cfg)
    bridge = subprocess.Popen([str(Path(__file__).resolve().parents[1]/'object_pose'),
        cfg['world'],cfg['object_name']], stdout=subprocess.PIPE, text=True, bufsize=1)
    reader = threading.Thread(target=node.read_poses,args=(bridge.stdout,),daemon=True)
    reader.start()
    try:
        node.wait(3)
        if args.mode == 'diagnose':
            print(json.dumps({'joint_names': JOINTS, 'state': node.state,
                'object_xyz_base': node.xyz, 'sim_time': node.sim_now(), 'ik_available': node.ik.wait_for_service(timeout_sec=2)}, indent=2))
            if node.state is None:
                raise RuntimeError('Joint-state connection unavailable')
            return
        if args.mode == 'scan':
            node.ready()
            grasp = [a+b for a,b in zip(node.xyz,cfg['grasp_offset_xyz'])]
            original_pitch = cfg['ik_pitch']
            print('IK comparison only: no joint commands, no configuration changes.', flush=True)
            print(f'object_xyz={node.xyz}; grasp_xyz={grasp}; lift={cfg["lift_height"]} m', flush=True)
            print('PASS means endpoint IK only; finger clearance, grasp alignment and paths are unverified.', flush=True)
            try:
                for pitch in dict.fromkeys([original_pitch, 3.05, 2.95, 2.85]):
                    cfg['ik_pitch'] = pitch
                    seed = list(node.state)
                    passed = True
                    print(f'\npitch={pitch:.6f} rad ({math.degrees(pitch):.2f} deg)', flush=True)
                    for label, dz, grip in [('approach',cfg['approach_height'],cfg['gripper_open']),
                                           ('grasp',0,cfg['gripper_open']),
                                           ('close',0,cfg['gripper_closed']),
                                           ('lift',cfg['lift_height'],cfg['gripper_closed'])]:
                        target = [*grasp[:2],grasp[2]+dz]
                        try:
                            seed = node.solve(target,seed,grip)
                            margins = [min(q-lo,hi-q) for q,(lo,hi) in zip(seed,cfg['joint_limits'])]
                            if min(margins) < 0:
                                raise ValueError('Configured joint limits exceeded')
                            print(f'  {label}: OK; min_limit_margin={min(margins):.4f} rad; joints={seed}', flush=True)
                        except (RuntimeError, ValueError, KeyError) as error:
                            passed = False
                            print(f'  {label}: FAIL; {error}', flush=True)
                            break
                    print('  result: '+('PASS' if passed else 'FAIL'), flush=True)
            finally:
                cfg['ik_pitch'] = original_pitch
            return
        if args.mode == 'plan':
            node.ready()
            grasp = [a+b for a,b in zip(node.xyz,cfg['grasp_offset_xyz'])]
            seed = list(node.state)
            for label, dz, grip in [('approach',cfg['approach_height'],cfg['gripper_open']), ('grasp',0,cfg['gripper_open']), ('close',0,cfg['gripper_closed']), ('lift',cfg['lift_height'],cfg['gripper_closed'])]:
                seed = node.solve([*grasp[:2],grasp[2]+dz],seed,grip)
                print(label, seed, flush=True)
            return
        if not cfg['calibrated'] and args.mode != 'trial':
            raise RuntimeError(f'Calibrate grasp geometry and joint limits, then set calibrated=true in {args.config}')
        node.ready()
        rng = random.Random(args.seed)
        policy = None
        if args.mode == 'infer':
            if not args.model_path:
                parser.error('infer requires --model-path')
            import torch
            from lerobot.common.policies.act.modeling_act import ACTPolicy
            policy = ACTPolicy.from_pretrained(args.model_path).to('cpu').eval()
        for episode in range(args.episodes):
            node.phase(cfg['home'], 3, label='home')
            position = list(cfg['spawn_world_xyz'])
            for i in range(2):
                position[i] += rng.uniform(-cfg['random_xy_half_range'][i],cfg['random_xy_half_range'][i])
            ign_service(cfg['world'], 'set_pose', 'ignition.msgs.Pose',
                f'name: "{cfg["object_name"]}" position {{x: {position[0]} y: {position[1]} z: {position[2]}}} orientation {{w: 1}}')
            node.wait(1)
            node.ready()
            if args.wait_for_recording:
                input('Home and cube reset complete. Start screen recording, then press Enter to begin: ')
                node.wait(0.5)
                node.ready()
            initial = list(node.xyz)
            path = Path(args.output) / f'{args.mode}_{time.time_ns()}'
            path.mkdir(parents=True, exist_ok=False)
            (path/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
            frames = []
            with (path/'frames.jsonl').open('w') as stream:
                try:
                    if policy is None:
                        grasp = [a+b for a,b in zip(initial, cfg['grasp_offset_xyz'])]
                        approach = [*grasp[:2],grasp[2]+cfg['approach_height']]
                        lift = [*grasp[:2],grasp[2]+cfg['lift_height']]
                        seed = list(node.state)
                        # Resolve all waypoints before moving toward the object.
                        targets = []
                        for point, grip in [(approach,cfg['gripper_open']), (grasp,cfg['gripper_open']),
                                            (grasp,cfg['gripper_closed']), (lift,cfg['gripper_closed'])]:
                            seed = node.solve(point,seed,grip)
                            targets.append(seed)
                        for label, target, seconds in zip(['approach','grasp','close','lift'], targets,[2,2,6,2]):
                            frames.extend(node.phase(target,seconds,stream,label=label))
                            if label == 'close':
                                # Keep the final target fixed instead of starting another ramp.
                                frames.extend(node.phase(target, cfg.get('close_settle_seconds', 3.0),
                                    stream, label='close_settle', ramp=False))
                        frames.extend(node.phase(targets[-1],1.5,stream,label="hold"))
                    else:
                        policy.reset()
                        deadline = node.sim_now()
                        for _ in range(round(args.seconds*FPS)):
                            node.ready()
                            obs = {'observation.state': list(node.state), 'observation.environment_state': list(node.xyz)}
                            with torch.no_grad():
                                action = policy.select_action({k:torch.tensor([v],dtype=torch.float32) for k,v in obs.items()}).flatten().tolist()
                            frame = validate_frame({**obs,'action': action,'timestamp':node.sim_now(),
                                'joint_names':list(JOINTS),'object_frame':'crane_plus_base'})
                            node.publish(action)
                            stream.write(json.dumps(frame)+'\n')
                            frames.append(frame)
                            deadline += 1/FPS
                            if node.sim_now() > deadline+0.015:
                                raise RuntimeError('Inference missed 20 Hz deadline')
                            node.wait_sim_until(deadline)
                    heights = [f['observation.environment_state'][2] for f in frames]
                    lift_threshold = cfg.get('success_lift_height', 0.04)
                    hold_seconds = cfg.get('success_hold_seconds', 1.0)
                    success = held_above(heights, initial[2], lift=lift_threshold, hold=hold_seconds)
                    tail = heights[-math.ceil(FPS*hold_seconds):]
                    result = {'success':success,'fps':FPS,'initial_xyz':initial,'seed':args.seed,'mode':args.mode,
                        'success_criteria':{'lift_height_m':lift_threshold,'hold_seconds':hold_seconds},
                        'max_object_lift_m':max(heights)-initial[2],
                        'final_window_min_object_lift_m':min(tail)-initial[2]}
                except Exception as error:
                    result = {'success':False,'fps':FPS,'error':str(error),'mode':args.mode}
                    (path/'result.json').write_text(json.dumps(result,indent=2))
                    raise
            (path/'result.json').write_text(json.dumps(result,indent=2))
            print(path, result)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        bridge.terminate()
        try:
            bridge.wait(timeout=3)
        except subprocess.TimeoutExpired:
            bridge.kill()
            bridge.wait()

if __name__ == '__main__':
    main()
