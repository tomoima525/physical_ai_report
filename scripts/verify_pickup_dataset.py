#!/usr/bin/env python3
"""Verify exported data against every source frame and load an ACT training batch."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/lerobot/pickup')
    parser.add_argument('--report', help='Optional JSON report path')
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(2)
    root = Path(args.root).resolve()
    info = json.loads((root/'meta/info.json').read_text())
    manifest = json.loads((root/'meta/conversion_manifest.json').read_text())
    metadata_episodes = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines()]
    assert info['fps'] == manifest['fps'] == 20
    assert info['total_episodes'] == len(manifest['episodes']) == len(metadata_episodes)
    assert info['total_frames'] == manifest['total_frames']
    expected_features = {'observation.state':5, 'observation.environment_state':3, 'action':5}
    for key, size in expected_features.items():
        assert info['features'][key]['shape'] == [size]
        assert info['features'][key]['dtype'] == 'float32'
    count = 0
    errors = {key:0.0 for key in expected_features}
    sources = []
    for i, entry in enumerate(manifest['episodes']):
        source = Path(entry['source_episode'])
        assert entry['dataset_episode_index'] == i
        assert source.name not in manifest['excluded_episode_names']
        raw = (source/'frames.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['frames_sha256'], source
        rows = [json.loads(line) for line in raw.splitlines()]
        parquet_path = root/info['data_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i)
        table = pq.read_table(parquet_path)
        n = len(rows)
        assert n == len(table) == entry['frame_count'] == metadata_episodes[i]['length']
        for key, size in expected_features.items():
            assert str(table.schema.field(key).type.value_type) == 'float'
            actual = np.asarray(table[key].to_pylist())
            expected = np.asarray([row[key] for row in rows])
            assert actual.shape == (n,size)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
            errors[key] = max(errors[key],float(np.max(np.abs(actual-expected))))
        np.testing.assert_array_equal(table['episode_index'].to_pylist(),[i]*n)
        np.testing.assert_array_equal(table['frame_index'].to_pylist(),np.arange(n))
        np.testing.assert_array_equal(table['index'].to_pylist(),np.arange(count,count+n))
        np.testing.assert_array_equal(table['task_index'].to_pylist(),[0]*n)
        np.testing.assert_allclose(table['timestamp'].to_pylist(),np.arange(n)/20,atol=1e-5)
        assert json.loads((source/'result.json').read_text())['success'] is True
        sources.append(source.name)
        count += n
    assert count == info['total_frames']
    dataset = LeRobotDataset('crane_plus_pickup',root=root,delta_timestamps={
        'observation.state':[0.0], 'observation.environment_state':[0.0],
        'action':[i/20 for i in range(16)]})
    assert len(dataset) == count
    # Read episode boundaries through the same loader used by training.
    boundary = 0
    for entry in manifest['episodes']:
        for index in [boundary,boundary+entry['frame_count']-1]:
            frame = dataset[index]
            assert tuple(frame['observation.state'].shape) == (1,5)
            assert tuple(frame['observation.environment_state'].shape) == (1,3)
            assert tuple(frame['action'].shape) == (16,5)
            assert torch.isfinite(frame['action']).all()
        boundary += entry['frame_count']
    batch = next(iter(torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)))
    shapes = {key:list(batch[key].shape) for key in expected_features}
    assert shapes == {'observation.state':[2,1,5], 'observation.environment_state':[2,1,3], 'action':[2,16,5]}
    report = {'verified':True,'dataset_root':str(root),'episodes':len(sources),'frames':count,
              'fps':info['fps'],'all_source_frames_match':True,'max_float_conversion_error':errors,
              'batch_shapes':shapes,'episode_boundaries_loaded':len(sources)*2,
              'excluded_episode_names':manifest['excluded_episode_names'],
              'notes':['Parquet timestamps are rebuilt on a 20 Hz grid after raw interval validation.',
                       'Metadata currently places all episodes in train; no held-out split or training run was evaluated.']}
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
