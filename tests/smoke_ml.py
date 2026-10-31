"""Synthetic-only export and ACT forward/save/load check; never a grasp result."""
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE']='1'
from pickup.core import JOINTS
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from configs.crane_features import features

torch.set_num_threads(2)
with tempfile.TemporaryDirectory(prefix='pickup_synthetic_') as tmp:
    root=Path(tmp)
    for e in range(2):
        ep=root/'episodes'/str(e)
        ep.mkdir(parents=True)
        (ep/'result.json').write_text(json.dumps({'success':True,'fps':20,'synthetic':True}))
        frames=[]
        for i in range(40):
            frames.append({'timestamp':i/20,'joint_names':list(JOINTS),'object_frame':'crane_plus_base',
                'observation.state':[math.sin(i/20+j)*.1 for j in range(5)],
                'observation.environment_state':[.15+e*.01,0,.03+i*.001],
                'action':[math.sin((i+1)/20+j)*.1 for j in range(5)]})
        (ep/'frames.jsonl').write_text('\n'.join(json.dumps(f) for f in frames))
    subprocess.run([sys.executable,'scripts/pickup_dataset.py','--episodes',str(root/'episodes'),'--output',str(root/'dataset')],check=True)
    ds=LeRobotDataset('crane_plus_pickup',root=root/'dataset',delta_timestamps={
        'observation.state':[0.], 'observation.environment_state':[0.], 'action':[i/20 for i in range(16)]})
    ft=dataset_to_policy_features(features)
    cfg=ACTConfig(input_features={k:v for k,v in ft.items() if v.type is not FeatureType.ACTION},
        output_features={k:v for k,v in ft.items() if v.type is FeatureType.ACTION},
        chunk_size=16,n_action_steps=1,dim_model=64,n_heads=4,dim_feedforward=128,
        n_encoder_layers=1,n_decoder_layers=1,n_vae_encoder_layers=1)
    meta=LeRobotDatasetMetadata('crane_plus_pickup',root=root/'dataset')
    model=ACTPolicy(cfg,dataset_stats=meta.stats)
    batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=2)))
    for key in ['observation.state','observation.environment_state']:
        batch[key]=batch[key][:,-1,:]
    model.train()
    loss,_=model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    model.save_pretrained(root/'policy')
    restored=ACTPolicy.from_pretrained(root/'policy').eval()
    restored.reset()
    with torch.no_grad():
        action=restored.select_action({k:batch[k] for k in ft if k!='action'})
    assert action.shape==(2,5) and torch.isfinite(action).all()
    print(f'SYNTHETIC smoke passed: {len(ds)} frames, loss={loss.item():.4f}, action={tuple(action.shape)}')
