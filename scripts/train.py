# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Tatsuya Kamijo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Tatsuya Kamijo (based on HuggingFace LeRobot lerobot/examples/3_train_policy.py)

"""ACT training with episode-level validation and train-only normalization."""
import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pickup.training import load_bound_split


def load_config_py(path):
    spec = importlib.util.spec_from_file_location('crane_features',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/train_config.yaml')
    parser.add_argument('--validate-only',action='store_true',help='Load split and batches without training')
    args=parser.parse_args()
    import yaml
    train_cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(train_cfg['dataset_root'])
    split, indices=load_bound_split(root,train_cfg['split_path'])
    output=Path(train_cfg['output_directory'])
    for key in ['training_steps','batch_size','chunk_size','n_action_steps','log_freq','eval_freq']:
        if train_cfg[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not args.validate_only and output.exists():
        raise FileExistsError(f'Output exists; use a new output_directory: {output}')
    os.environ.setdefault('WANDB_MODE','offline')
    import numpy as np
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.compute_stats import aggregate_stats
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType, NormalizationMode
    seed=train_cfg.get('seed',20261031)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(train_cfg.get('torch_num_threads',2))
    device=torch.device(train_cfg['device'])
    dataset=LeRobotDataset(train_cfg['repo_id'],root=root,delta_timestamps=train_cfg['delta_timestamps'])
    expected=[i/dataset.fps for i in range(train_cfg['chunk_size'])]
    if train_cfg['delta_timestamps']['action'] != expected:
        raise ValueError('Action chunk timestamps must match dataset fps and chunk_size')
    train_subset=torch.utils.data.Subset(dataset,indices['train'])
    validation_subset=torch.utils.data.Subset(dataset,indices['validation'])
    train_loader=torch.utils.data.DataLoader(train_subset,batch_size=train_cfg['batch_size'],
        shuffle=True,num_workers=train_cfg['num_workers'],drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    validation_loader=torch.utils.data.DataLoader(validation_subset,batch_size=train_cfg['batch_size'],
        shuffle=False,num_workers=0,drop_last=False)
    if not len(train_loader):
        raise ValueError('Training split smaller than batch_size')
    # Never use dataset.meta.stats here: it includes validation episodes.
    stats=aggregate_stats([dataset.meta.episodes_stats[i] for i in split['train_episodes']])

    def prepare(batch):
        batch={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
        for key,size in [('observation.state',5),('observation.environment_state',3)]:
            if batch[key].ndim != 3 or batch[key].shape[1:] != (1,size):
                raise ValueError(f'Unexpected {key} shape: {tuple(batch[key].shape)}')
            batch[key]=batch[key][:,-1,:]
        return batch

    split_summary={'train_episodes':split['train_episodes'],'validation_episodes':split['validation_episodes'],
        'train_frames':len(train_subset),'validation_frames':len(validation_subset),
        'normalization_stats_from':'train episodes only','model':train_cfg.get('model',{})}
    # Exercise both selected subsets and every episode's last action chunk.
    train_batch=prepare(next(iter(train_loader)))
    val_batch=prepare(next(iter(validation_loader)))
    offset=0
    for episode in dataset.meta.episodes.values():
        end=offset+episode['length']-1
        frame=dataset[end]
        assert int(frame['episode_index']) == episode['episode_index']
        assert tuple(frame['action'].shape) == (train_cfg['chunk_size'],5)
        assert bool(frame['action_is_pad'][-1])
        offset=end+1
    features=dataset_to_policy_features(load_config_py(train_cfg['config_path']).features)
    act_cfg=ACTConfig(input_features={k:v for k,v in features.items() if v.type is not FeatureType.ACTION},
        output_features={k:v for k,v in features.items() if v.type is FeatureType.ACTION},
        chunk_size=train_cfg['chunk_size'],n_action_steps=train_cfg['n_action_steps'],device=str(device),
        optimizer_lr=train_cfg['learning_rate'],optimizer_lr_backbone=train_cfg['learning_rate'],
        optimizer_weight_decay=0.0, **train_cfg.get('model',{}))
    normalize_environment=train_cfg.get('normalize_environment',False)
    if not isinstance(normalize_environment,bool):
        raise ValueError('normalize_environment must be a boolean')
    if normalize_environment:
        act_cfg.normalization_mapping[FeatureType.ENV]=NormalizationMode.MEAN_STD
    split_summary['normalization_mapping']={str(k.value if hasattr(k,'value') else k):
        v.value for k,v in act_cfg.normalization_mapping.items()}
    split_summary['prepared_train_batch_shapes']={k:list(train_batch[k].shape)
        for k in ['observation.state','observation.environment_state','action']}
    split_summary.update({'training_steps':train_cfg['training_steps'],
                          'output_directory':str(output),'optimizer_lr':act_cfg.optimizer_lr})
    print(json.dumps(split_summary,indent=2),flush=True)
    policy=ACTPolicy(act_cfg,dataset_stats=stats).to(device)
    if normalize_environment:
        with torch.no_grad():
            normalized=policy.normalize_inputs({'observation.environment_state':train_batch['observation.environment_state']})
            env_stats=stats['observation.environment_state']
            mean=torch.as_tensor(env_stats['mean'],device=device,dtype=torch.float32)
            std=torch.as_tensor(env_stats['std'],device=device,dtype=torch.float32)
            torch.testing.assert_close(normalized['observation.environment_state'],
                (train_batch['observation.environment_state']-mean)/(std+1e-8))
    if args.validate_only:
        print('Split/batch validation passed. No training or checkpoint creation.',flush=True)
        return
    optimizer=torch.optim.Adam(policy.parameters(),lr=act_cfg.optimizer_lr,weight_decay=act_cfg.optimizer_weight_decay)
    output.mkdir(parents=True,exist_ok=False)
    run={'config_file':args.config,'config':train_cfg,'split':split,'split_summary':split_summary,
         'validation_metric':'first action MAE in radians; no ground-truth actions passed into policy',
         'purpose':train_cfg.get('purpose','training')}
    (output/'run_config.json').write_text(json.dumps(run,indent=2)+'\n')
    import wandb
    wb=wandb.init(project='pai_crane_gazebo',config=run,dir=str(output),
                  mode=os.environ.get('WANDB_MODE','offline'))

    def evaluate():
        policy.eval()
        totals=torch.zeros(5,device=device)
        count=0
        with torch.no_grad():
            for raw in validation_loader:
                batch=prepare(raw)
                # Reset the action queue per independent observation, and omit target actions.
                policy.reset()
                pred=policy.select_action({k:batch[k] for k in features if k != 'action'})
                errors=(pred-batch['action'][:,0,:]).abs()
                if not torch.isfinite(errors).all():
                    raise ValueError('Nonfinite validation prediction')
                totals+=errors.sum(dim=0)
                count+=len(pred)
        per_joint=(totals/count).cpu().tolist()
        policy.reset()
        return {'validation_first_action_mae_rad':sum(per_joint)/len(per_joint),
                'validation_per_joint_mae_rad':per_joint,'validation_frame_count':count}

    metrics_path=output/'metrics.jsonl'
    def record(row):
        print(json.dumps(row),flush=True)
        with metrics_path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        wb.log(row,step=row['step'])

    try:
        record({'step':0,**evaluate()})
        step=0
        while step < train_cfg['training_steps']:
            for raw in train_loader:
                policy.train()
                batch=prepare(raw)
                optimizer.zero_grad(set_to_none=True)
                loss,loss_parts=policy(batch)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step()
                step+=1
                row={'step':step}
                if step % train_cfg['log_freq']==0 or step==1:
                    row.update({'train_loss':float(loss.detach().cpu()),'gradient_norm':float(grad_norm),
                                **{f'train_{k}':v for k,v in loss_parts.items()}})
                if step % train_cfg['eval_freq']==0 or step==train_cfg['training_steps']:
                    row.update(evaluate())
                if len(row)>1:
                    record(row)
                if step>=train_cfg['training_steps']:
                    break
        policy.save_pretrained(output)
        # Check the saved model can perform the same action selection used by inference.
        loaded=ACTPolicy.from_pretrained(output).to(device).eval()
        loaded.reset()
        if normalize_environment:
            assert loaded.config.normalization_mapping[FeatureType.ENV] is NormalizationMode.MEAN_STD
            torch.testing.assert_close(
                loaded.normalize_inputs({'observation.environment_state':val_batch['observation.environment_state']})['observation.environment_state'],
                policy.normalize_inputs({'observation.environment_state':val_batch['observation.environment_state']})['observation.environment_state'])
        with torch.no_grad():
            action=loaded.select_action({k:val_batch[k] for k in features if k!='action'})
        assert action.shape==(len(val_batch['action']),5) and torch.isfinite(action).all()
        summary={'completed_steps':step,'checkpoint_reloaded':True,'action_shape':list(action.shape),
                 'purpose':train_cfg.get('purpose','training')}
        (output/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f'Training complete; model saved and reloaded: {output}',flush=True)
    finally:
        wb.finish()


if __name__=='__main__':
    main()
