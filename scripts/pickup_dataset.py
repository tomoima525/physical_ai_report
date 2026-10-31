#!/usr/bin/env python3
"""Convert successful synchronized JSONL episodes to pinned LeRobot v2.1."""
import argparse
import json
import hashlib
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
        # Avoid rejecting the exact 15 ms tolerance boundary due to float subtraction.
        if abs(b['timestamp'] - a['timestamp'] - 1 / FPS) > 0.015 + 1e-9:
            raise ValueError(f'{path}: irregular sample interval; recollect episode')
    return frames

def select_episodes(root, excluded=()):
    paths = sorted(p.parent for p in Path(root).glob('*/result.json'))
    excluded = set(excluded)
    unknown = excluded - {p.name for p in paths}
    if unknown:
        raise ValueError(f'Excluded episode not found: {sorted(unknown)}')
    selected = []
    for path in paths:
        if path.name in excluded:
            continue
        frames = read_episode(path)
        if frames:
            selected.append((path, frames))
    if not selected:
        raise ValueError('No successful episodes selected; collect and validate physical grasps first')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default='data/episodes')
    parser.add_argument('--output', default='data/lerobot/pickup')
    parser.add_argument('--exclude-episode', action='append', default=[],
        help='Exact episode directory name to exclude; may be repeated')
    parser.add_argument('--validate-only', action='store_true',
        help='Validate selected raw episodes without creating a dataset or importing ML libraries')
    args = parser.parse_args()
    # Fail on any unreviewed invalid episode; exclusions must be explicit.
    selected = select_episodes(args.episodes, args.exclude_episode)
    frame_count = sum(len(frames) for _, frames in selected)
    print(f'Selected {len(selected)} episodes, {frame_count} frames; excluded: {args.exclude_episode}')
    if args.validate_only:
        print('Validation passed; no dataset was created.')
        return
    if Path(args.output).exists():
        raise FileExistsError(f'Output already exists; choose another --output: {args.output}')
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from configs.crane_features import features
    import numpy as np
    dataset = LeRobotDataset.create(repo_id='crane_plus_pickup', fps=FPS,
        root=Path(args.output).resolve(), robot_type='crane_plus', features=features, use_videos=False)
    for source_path, episode in selected:
        for row in episode:
            dataset.add_frame({**{k: np.asarray(row[k], dtype=np.float32) for k in features},
                               'task': 'Lift the cube by 4 cm and hold for 1 second'})
        dataset.save_episode()
    manifest = {
        'source_root': str(Path(args.episodes)),
        'excluded_episode_names': args.exclude_episode,
        'fps': FPS, 'total_frames': frame_count,
        'episodes': [{'dataset_episode_index': i, 'source_episode': str(path), 'frame_count': len(frames),
                      'frames_sha256': hashlib.sha256((path/'frames.jsonl').read_bytes()).hexdigest()}
                     for i, (path, frames) in enumerate(selected)],
    }
    (Path(dataset.root)/'meta'/'conversion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved {len(selected)} episodes, {frame_count} frames to {dataset.root}')

if __name__ == '__main__':
    main()
