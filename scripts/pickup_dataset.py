#!/usr/bin/env python3
"""Convert successful synchronized JSONL episodes to pinned LeRobot v2.1."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pickup.core import FPS, validate_frame

def read_episode(path):
    summary = json.loads((path / 'result.json').read_text())
    if summary.get('success') is not True:
        return []
    if summary.get('fps') != FPS:
        raise ValueError(f'{path}: fps mismatch')
    frames = [validate_frame(json.loads(line)) for line in (path / 'frames.jsonl').read_text().splitlines()]
    if len(frames) < 2:
        raise ValueError(f'{path}: episode too short')
    for a, b in zip(frames, frames[1:]):
        if abs(b['timestamp'] - a['timestamp'] - 1 / FPS) > 0.015:
            raise ValueError(f'{path}: irregular sample interval; recollect episode')
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default='data/episodes')
    parser.add_argument('--output', default='data/lerobot/pickup')
    args = parser.parse_args()
    # Validate everything before creating the dataset. Never overwrite a dataset.
    episodes = [read_episode(p.parent) for p in sorted(Path(args.episodes).glob('*/result.json'))]
    episodes = [ep for ep in episodes if ep]
    if not episodes:
        raise ValueError('No successful episodes; collect and validate physical grasps first')
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from configs.crane_features import features
    import numpy as np
    dataset = LeRobotDataset.create(repo_id='crane_plus_pickup', fps=FPS,
        root=Path(args.output).resolve(), robot_type='crane_plus', features=features, use_videos=False)
    for episode in episodes:
        for row in episode:
            dataset.add_frame({**{k: np.asarray(row[k], dtype=np.float32) for k in features},
                               'task': 'Lift the cube by 5 cm and hold for 1 second'})
        dataset.save_episode()
    print(f'Saved {len(episodes)} episodes to {dataset.root}')

if __name__ == '__main__':
    main()
