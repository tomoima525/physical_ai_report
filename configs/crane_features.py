from pickup.core import JOINTS
features = {
    "observation.state": {"dtype": "float32", "shape": (5,), "names": list(JOINTS)},
    "observation.environment_state": {"dtype": "float32", "shape": (3,), "names": ["object_x", "object_y", "object_z"]},
    "action": {"dtype": "float32", "shape": (5,), "names": list(JOINTS)},
}
robot_type = "crane_plus"
repo_id = "crane_plus_pickup"
root = "data/lerobot/pickup"
task_name = "Lift the cube by 4 cm and hold for 1 second"
