#!/usr/bin/env python3
"""Read-only offline checkpoint review; never publishes robot commands."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pickup.training import load_bound_split


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='checkpoints/pickup_act_smoke')
    parser.add_argument('--report')
    args=parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('HF_DATASETS_OFFLINE','1')
    import numpy as np
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.compute_stats import aggregate_stats
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    torch.set_num_threads(2)
    checkpoint=Path(args.checkpoint)
    run=json.loads((checkpoint/'run_config.json').read_text())
    summary=json.loads((checkpoint/'training_summary.json').read_text())
    cfg=run['config']
    split,indices=load_bound_split(cfg['dataset_root'],cfg['split_path'])
    assert split==run['split']
    assert summary['completed_steps']==cfg['training_steps'] and summary['checkpoint_reloaded']
    assert not set(split['train_episodes']) & set(split['validation_episodes'])
    manifest=json.loads((Path(cfg['dataset_root'])/'meta/conversion_manifest.json').read_text())
    frames={}
    for entry in manifest['episodes']:
        source=Path(entry['source_episode'])
        raw=(source/'frames.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==entry['frames_sha256']
        frames[entry['dataset_episode_index']]=[json.loads(line) for line in raw.splitlines()]
    train=[f for ep in split['train_episodes'] for f in frames[ep]]
    val=[f for ep in split['validation_episodes'] for f in frames[ep]]
    mean=np.asarray([f['action'] for f in train],dtype=np.float64).mean(axis=0)
    policy=ACTPolicy.from_pretrained(checkpoint).to('cpu').eval()
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    meta=LeRobotDatasetMetadata(cfg['repo_id'],root=Path(cfg['dataset_root']))
    stats=aggregate_stats([meta.episodes_stats[i] for i in split['train_episodes']])
    for module,key in [(policy.normalize_inputs,'observation.state'),
                       (policy.normalize_targets,'action'),(policy.unnormalize_outputs,'action')]:
        buf=getattr(module,'buffer_'+key.replace('.','_'))
        for statistic in ['mean','std']:
            torch.testing.assert_close(buf[statistic],torch.tensor(stats[key][statistic],dtype=torch.float32))
    environment_normalized=cfg.get('normalize_environment',False)
    if environment_normalized:
        from lerobot.configs.types import FeatureType, NormalizationMode
        assert policy.config.normalization_mapping[FeatureType.ENV] is NormalizationMode.MEAN_STD
        buf=policy.normalize_inputs.buffer_observation_environment_state
        for statistic in ['mean','std']:
            torch.testing.assert_close(buf[statistic],
                torch.tensor(stats['observation.environment_state'][statistic],dtype=torch.float32))
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(val),cfg['batch_size']):
            rows=val[start:start+cfg['batch_size']]
            obs={k:torch.tensor([f[k] for f in rows],dtype=torch.float32)
                 for k in ['observation.state','observation.environment_state']}
            policy.reset()
            predicted=policy.select_action(obs)
            assert predicted.shape==(len(rows),5) and torch.isfinite(predicted).all()
            predictions.extend(predicted.tolist())
    pred=np.asarray(predictions,dtype=np.float64)
    target=np.asarray([f['action'] for f in val],dtype=np.float64)
    state=np.asarray([f['observation.state'] for f in val],dtype=np.float64)
    def errors(estimate,indices=None):
        x=np.abs(estimate-target)
        if indices is not None:
            x=x[indices]
        per_joint=x.mean(axis=0).tolist()
        return {'mae_rad':sum(per_joint)/5,'per_joint_mae_rad':per_joint}
    final=errors(pred)
    metrics=[json.loads(line) for line in (checkpoint/'metrics.jsonl').read_text().splitlines()]
    logged=metrics[-1]['validation_first_action_mae_rad']
    assert math.isclose(final['mae_rad'],logged,abs_tol=1e-6)
    phase_results={}
    for phase in dict.fromkeys(f['phase'] for f in val):
        selected=[i for i,f in enumerate(val) if f['phase']==phase]
        phase_results[phase]={'frame_count':len(selected),'model':errors(pred,selected),
                              'current_joint_baseline':errors(state,selected)}
    lo=np.asarray([v[0] for v in json.loads((Path(manifest['episodes'][0]['source_episode'])/'config.json').read_text())['joint_limits']])
    hi=np.asarray([v[1] for v in json.loads((Path(manifest['episodes'][0]['source_episode'])/'config.json').read_text())['joint_limits']])
    report={'checkpoint':str(checkpoint),'model_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        'completed_steps':summary['completed_steps'],'train_episodes':split['train_episodes'],
        'validation_episodes':split['validation_episodes'],'train_frames':len(train),'validation_frames':len(val),
        'environment_normalized':environment_normalized,'normalization_matches_train_only_stats':True,'checkpoint_parameters_finite':True,
        'validation_before_mae_rad':metrics[0]['validation_first_action_mae_rad'],'validation_after':final,
        'logged_metric_matches_recomputed':True,'current_joint_baseline':errors(state),
        'train_mean_action_baseline':errors(np.broadcast_to(mean,target.shape)),
        'phase_results':phase_results,'predictions_outside_joint_limits':int(((pred<lo)|(pred>hi)).any(axis=1).sum()),
        'out_of_bounds_per_joint':((pred<lo)|(pred>hi)).sum(axis=0).tolist(),
        'predicted_joint_ranges':list(zip(pred.min(axis=0).tolist(),pred.max(axis=0).tolist())),
        'configured_joint_limits':list(zip(lo.tolist(),hi.tolist())),
        'out_of_bounds_by_phase':{phase:int(((pred<lo)|(pred>hi))[[i for i,f in enumerate(val) if f['phase']==phase]].any(axis=1).sum()) for phase in phase_results},
        'limitations':['Offline independent-observation first-action prediction only.',
                      'Current-joint baseline issues no movement and cannot perform the task.',
                      'No simulator inference, grasp success rate, robustness or generalization test was performed.']}
    if args.report:
        Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
