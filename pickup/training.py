"""Dependency-free validation for episode-level training splits."""
import hashlib
import json
from pathlib import Path


def split_frame_indices(episodes, split):
    train, validation = split['train_episodes'], split['validation_episodes']
    ids = [row['episode_index'] for row in episodes]
    if ids != list(range(len(ids))):
        raise ValueError('Expected consecutive dataset episode IDs')
    for name, selected in [('train', train), ('validation', validation)]:
        if not selected or len(set(selected)) != len(selected):
            raise ValueError(f'{name} split is empty or contains duplicates')
        if any(type(i) is not int or i not in ids for i in selected):
            raise ValueError(f'{name} contains unknown episode IDs')
    if set(train) & set(validation):
        raise ValueError('Train and validation episodes overlap')
    if set(train) | set(validation) != set(ids):
        raise ValueError('Split must account for every dataset episode')
    ranges = {}
    offset = 0
    for episode in episodes:
        n = episode['length']
        if n <= 0:
            raise ValueError('Empty episode')
        ranges[episode['episode_index']] = range(offset, offset+n)
        offset += n
    return {name: [index for ep in selected for index in ranges[ep]]
            for name, selected in [('train',train),('validation',validation)]}


def load_bound_split(root, split_path):
    root = Path(root)
    split = json.loads(Path(split_path).read_text())
    manifest = (root/'meta/conversion_manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != split['conversion_manifest_sha256']:
        raise ValueError('Dataset manifest changed; review and regenerate the split')
    episodes = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines()]
    return split, split_frame_indices(episodes,split)
