from configs.crane_features import features as original_features
features = {k: dict(v) for k, v in original_features.items()}
features['observation.environment_state'] = {'dtype': 'float32', 'shape': (4,),
    'names': ['object_x', 'object_y', 'object_z', 'elapsed_sim_seconds']}
