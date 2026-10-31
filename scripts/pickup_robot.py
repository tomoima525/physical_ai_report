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

    def phase(self, target, seconds, stream=None):
        self.ready()
        start = list(self.state)
        frames = []
        deadline = self.sim_now()
        for i in range(round(seconds*FPS)):
            self.ready()
            alpha = min(1.0, (i+1)/(seconds*FPS))
            alpha = alpha*alpha*(3-2*alpha)
            action = [a+(b-a)*alpha for a,b in zip(start,target)]
            timestamp = self.sim_now()
            frame = validate_frame({'timestamp': timestamp, 'joint_names': list(JOINTS),
                'object_frame': 'crane_plus_base', 'observation.state': list(self.state),
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
        return frames


def ign_service(world, service, request_type, request):
    result = subprocess.run(['ign','service','-s',f'/world/{world}/{service}',
        '--reqtype',request_type,'--reptype','ignition.msgs.Boolean',
        '--timeout','5000','--req',request], capture_output=True, text=True, timeout=8, check=True)
    if 'data: true' not in result.stdout:
        raise RuntimeError(f'Gazebo {service} failed: {result.stdout} {result.stderr}')


def spawn(cfg):
    size = cfg['cube_size']
    xyz = ' '.join(str(v) for v in cfg['spawn_world_xyz'])
    sdf = f'''<sdf version="1.6"><model name="{cfg['object_name']}"><pose>{xyz} 0 0 0</pose>
<link name="body"><inertial><mass>0.02</mass><inertia><ixx>0.0000021</ixx><iyy>0.0000021</iyy><izz>0.0000021</izz></inertia></inertial>
<collision name="collision"><geometry><box><size>{size} {size} {size}</size></box></geometry>
<surface><friction><ode><mu>1.5</mu><mu2>1.5</mu2></ode></friction></surface></collision>
<visual name="visual"><geometry><box><size>{size} {size} {size}</size></box></geometry><material><diffuse>0.9 0.2 0.1 1</diffuse></material></visual>
</link></model></sdf>'''
    ign_service(cfg['world'], 'create', 'ignition.msgs.EntityFactory',
        'sdf: '+json.dumps(sdf)+' allow_renaming: false')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['diagnose','spawn','reset','plan','trial','collect','infer'])
    parser.add_argument('--config', default='configs/pickup.json')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='data/episodes')
    parser.add_argument('--model-path')
    parser.add_argument('--seconds', type=float, default=15)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.mode == 'spawn':
        spawn(cfg)
        return
    if args.mode == 'reset':
        x,y,z = cfg['spawn_world_xyz']
        ign_service(cfg['world'], 'set_pose', 'ignition.msgs.Pose',
            f'name: "{cfg["object_name"]}" position {{x: {x} y: {y} z: {z}}} orientation {{w: 1}}')
        return
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
        if args.mode == 'plan':
            node.ready()
            grasp = [a+b for a,b in zip(node.xyz,cfg['grasp_offset_xyz'])]
            seed = list(node.state)
            for label, dz, grip in [('approach',cfg['approach_height'],cfg['gripper_open']), ('grasp',0,cfg['gripper_open']), ('close',0,cfg['gripper_closed']), ('lift',cfg['lift_height'],cfg['gripper_closed'])]:
                seed = node.solve([*grasp[:2],grasp[2]+dz],seed,grip)
                print(label, seed, flush=True)
            return
        if not cfg['calibrated'] and args.mode != 'trial':
            raise RuntimeError('Calibrate grasp geometry and joint limits, then set calibrated=true in configs/pickup.json')
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
            node.phase(cfg['home'], 3)
            position = list(cfg['spawn_world_xyz'])
            for i in range(2):
                position[i] += rng.uniform(-cfg['random_xy_half_range'][i],cfg['random_xy_half_range'][i])
            ign_service(cfg['world'], 'set_pose', 'ignition.msgs.Pose',
                f'name: "{cfg["object_name"]}" position {{x: {position[0]} y: {position[1]} z: {position[2]}}} orientation {{w: 1}}')
            node.wait(1)
            node.ready()
            initial = list(node.xyz)
            path = Path(args.output) / f'{args.mode}_{time.time_ns()}'
            path.mkdir(parents=True, exist_ok=False)
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
                        for target, seconds in zip(targets,[2,2,1,2]):
                            frames.extend(node.phase(target,seconds,stream))
                        frames.extend(node.phase(targets[-1],1.5,stream))
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
                    success = held_above([f['observation.environment_state'][2] for f in frames],initial[2])
                    result = {'success':success,'fps':FPS,'initial_xyz':initial,'seed':args.seed,'mode':args.mode}
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
