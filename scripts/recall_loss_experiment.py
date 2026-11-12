"""Staged, validation-only Tversky+BCE study with matched corrected-seed controls."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key, value in {'HF_HUB_OFFLINE':'1', 'HF_HOME':str(ROOT/'outputs/huggingface_cache'),
                   'TORCH_HOME':str(ROOT/'outputs/torch_cache')}.items():
    os.environ.setdefault(key, value)
STUDY = ROOT/'outputs/experiments/recall_loss'
CHECKPOINTS = ROOT/'checkpoints/experiments/recall_loss'
THRESHOLDS = [round(.30+.05*i, 2) for i in range(11)]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for attempt in range(30):
        try:
            os.replace(temporary,path); return
        except PermissionError:
            time.sleep(.1)
    raise PermissionError(path)


def worker(seed):
    from src.config import Config
    from src.train import train
    import src.data as data
    output=STUDY/f'seed{seed}'
    checkpoint=CHECKPOINTS/f'seed{seed}'
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'history.csv').exists(), 'Refusing to overwrite any existing training history'
    jobs=json.loads((ROOT/'outputs/experiments/normalization_multiseed/jobs.json').read_text())
    reference=ROOT/jobs[f'whole_volume_seed{seed}']['output_dir']
    for name in ['split.json','normalization.json']:
        shutil.copy2(reference/name,output/name)
    splits=json.loads((output/'split.json').read_text())
    allowed=set(splits['train']+splits['val'])
    original_image,original_mask=data.read_image,data.read_mask
    def guarded_image(path):
        assert Path(path).parent.name in allowed, 'Attempt to read test image pixels'
        return original_image(path)
    def guarded_mask(path):
        assert Path(path).parent.name in allowed, 'Attempt to read test annotation'
        return original_mask(path)
    data.read_image,data.read_mask=guarded_image,guarded_mask
    start=time.perf_counter()
    path=train(Config(output_dir=output,checkpoint_dir=checkpoint,seed=seed,loss_name='tversky_bce'))
    import pandas as pd
    assert len(pd.read_csv(output/'history.csv'))==30
    save(output/'completed.json',dict(seed=seed,training_seconds=time.perf_counter()-start,
                                      checkpoint_sha256=fingerprint(path),epochs=30,loss_name='tversky_bce'))


def measure(seed, protocol):
    import numpy as np
    import pandas as pd
    import torch
    from src.config import Config
    from src.baseline_tuned import patient_files
    from src.data import read_mask
    from src.evaluate import predict_probabilities
    from src.model import load_model
    from src.utils import segmentation_metrics,device,seed_everything
    from scripts.recall_refinement_experiment import aggregate
    seed_everything(seed)
    csv_path=STUDY/f'seed{seed}_patient_metrics.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    splits=json.loads((ROOT/'outputs/split.json').read_text())
    frame=patient_files(Config(),splits['val'])
    jobs=json.loads((ROOT/'outputs/experiments/normalization_multiseed/jobs.json').read_text())
    paths={'reference':ROOT/jobs[f'whole_volume_seed{seed}']['checkpoint_dir'], 'candidate':CHECKPOINTS/f'seed{seed}'}
    rows=[]; dev=device()
    for role,path in paths.items():
        config=Config(checkpoint_dir=path,output_dir=STUDY/f'seed{seed}',prediction_threshold=.70)
        model,ckpt=load_model(config,dev)
        assert ckpt['splits']==splits and ckpt['seed']==seed
        assert ckpt.get('augmentation_seeding')=='worker_specific_persistent'
        assert ckpt.get('normalization_mode')=='whole_volume'
        if role=='reference':
            assert ckpt['checkpoint_sha256']==protocol['reference_checkpoints'][str(seed)]
        else:
            assert ckpt['loss_name']=='tversky_bce'
        for patient,group in frame.groupby('patient'):
            group=group.sort_values('index')
            signature=[[r.image,Path(r.image).stat().st_size,Path(r.image).stat().st_mtime_ns] for r in group.itertuples()]
            assert signature==ckpt['normalization'][patient]['signature']
            truth=np.stack([read_mask(r.mask) for r in group.itertuples()]).astype(bool)
            cache=STUDY/f'seed{seed}'/'probabilities'/role
            cache.mkdir(parents=True,exist_ok=True)
            prior=ROOT/f'outputs/experiments/recall_refinement/seed{seed}/probabilities/{patient}.npz'
            cached=prior if role=='reference' and prior.exists() else cache/f'{patient}.npz'
            if cached.exists():
                prob=np.load(cached)['probabilities']
            else:
                prob=predict_probabilities(model,group,ckpt['normalization'],config,dev)
                np.savez_compressed(cached,probabilities=prob)
            for threshold in THRESHOLDS:
                rows.append(dict(seed=seed,role=role,patient=patient,threshold=threshold,
                                 **segmentation_metrics(prob>=threshold,truth)))
        del model
        if dev.type=='cuda':torch.cuda.empty_cache()
    table=pd.DataFrame(rows)
    table.to_csv(csv_path,index=False)
    return table


def screening(table):
    from scripts.recall_refinement_experiment import aggregate
    reference=table[(table.role=='reference')&(table.threshold==.70)]
    base=aggregate(reference)
    choices=[]
    for threshold in THRESHOLDS:
        candidate=table[(table.role=='candidate')&(table.threshold==threshold)]
        metrics=aggregate(candidate)
        paired=reference[['patient','dice']].merge(candidate[['patient','dice']],on='patient',suffixes=('_reference','_candidate'))
        worst=float((paired.dice_candidate-paired.dice_reference).min())
        gains={key:metrics[key]-base[key] for key in base}
        passes=(gains['mean_dice']>=-.002 and gains['pooled_recall']>=.01
                and gains['pooled_precision']>=-.02 and gains['false_alarms']<=0
                and gains['missed_slices']<0 and worst>=-.05)
        choices.append(dict(threshold=threshold,eligible=passes,worst_patient_dice_change=worst,
                            metrics=metrics,changes=gains))
    eligible=[c for c in choices if c['eligible']]
    selected=max(eligible,key=lambda c:(c['metrics']['mean_dice'],c['metrics']['pooled_recall'],-abs(c['threshold']-.7))) if eligible else None
    return dict(reference=base,choices=choices,selected=selected,passes=bool(eligible))


def report(protocol, tables, screen):
    import pandas as pd
    from scripts.recall_refinement_experiment import aggregate
    table=pd.concat(tables,ignore_index=True)
    table.to_csv(STUDY/'validation_patient_metrics.csv',index=False)
    rows=[dict(seed=int(seed),role=role,threshold=float(threshold),**aggregate(group))
          for (seed,role,threshold),group in table.groupby(['seed','role','threshold'])]
    summary=pd.DataFrame(rows);summary.to_csv(STUDY/'validation_summary.csv',index=False)
    selected=screen['selected']['threshold'] if screen['passes'] else .70
    comparisons=[]
    for seed in sorted(table.seed.unique()):
        group=table[table.seed==seed]
        ref=aggregate(group[(group.role=='reference')&(group.threshold==.70)])
        cand=aggregate(group[(group.role=='candidate')&(group.threshold==selected)])
        comparisons.append(dict(seed=int(seed),reference=ref,candidate=cand,changes={k:cand[k]-ref[k] for k in ref}))
    confirmed=(len(comparisons)==3 and all(c['changes']['pooled_recall']>0 and c['changes']['mean_dice']>=-.005
               and c['changes']['false_alarms']<=0 and c['changes']['missed_slices']<=0 for c in comparisons)
               and sum(c['changes']['missed_slices'] for c in comparisons)<0
               and sum(c['changes']['mean_dice'] for c in comparisons)/3>=-.002
               and sum(c['changes']['pooled_recall'] for c in comparisons)/3>=.01)
    decision=dict(screen_passed=screen['passes'],selected_threshold=selected,seeds_completed=len(comparisons),
                  confirmed_across_three_seeds=confirmed,comparisons=comparisons,shipped_model_changed=False,
                  test_patients_evaluated=0,screen=screen,
                  decision='Promising development candidate; independent patient validation still required' if confirmed else
                  'No candidate met the staged confirmation requirements; retain shipped model')
    save(STUDY/'comparison.json',decision)
    lines=['# Recall-weighted loss experiment','',
           'Candidate training changes only Dice+BCE to Tversky+BCE (FP weight 0.3, FN weight 0.7). Architecture, initialization, optimizer, augmentations, natural slice sampling, whole-volume normalization, checkpoint selection at 0.5, and 30-epoch budget match the completed corrected-seeding controls. Neither inference refinement nor postprocessing is used in this loss comparison.','',
           'Seed 42 is screened first. The validation threshold grid and acceptance gates were frozen before training. A candidate must gain >=1 percentage point pooled recall, reduce completely missed positive slices, lose <=0.002 mean Dice and <=0.02 pooled precision, add no false-alarm slices, and lose <=0.05 Dice for any patient. A passing seed-42 threshold is frozen for seeds 43 and 44; otherwise the experiment stops rather than spending two additional training runs on a rejected candidate.','',
           f"**{decision['decision']}.** Screen passed: {screen['passes']}; completed seeds: {len(comparisons)}; selected candidate threshold: {selected:.2f}.",'',
           '| Seed | Model | Threshold | Mean Dice | Pooled precision | Pooled recall | False alarms | Missed slices |','|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in sorted(table.seed.unique()):
        for role,threshold in [('reference',.70),('candidate',.70)]+([('candidate',selected)] if selected!=.70 else []):
            r=summary[(summary.seed==seed)&(summary.role==role)&(summary.threshold==threshold)].iloc[0]
            lines.append(f'| {seed} | {role} | {threshold:.2f} | {r.mean_dice:.4f} | {r.pooled_precision:.2%} | {r.pooled_recall:.2%} | {int(r.false_alarms)}/{int(r.tumor_free_slices)} | {int(r.missed_slices)} |')
    lines += ['', '## Interpretation','',
              'This is development evidence on the same 14 reused validation patients. A negative screen does not prove the loss can never work; it means this prespecified candidate did not meet the project tradeoff requirements. A positive result would still need independent patient confirmation. The original 22 test patients contribute no labels or predictions to this experiment.','',
              'The shipped checkpoint, demo, 0.70 deployment threshold, classical baselines and all existing production numbers remain unchanged. Independent validation cannot be supplied by rerunning the existing test set.','',
              'Full threshold results are in validation_summary.csv; patient metrics, frozen protocol, seed histories, checkpoint metadata, and decisions are retained alongside this report.','']
    for p,h in protocol['protected_sha256'].items():assert fingerprint(p)==h,p
    (STUDY/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    save(STUDY/'completed.json',dict(**decision,protected_artifacts_unchanged=True))
    print(json.dumps(decision,indent=2),flush=True)


def run():
    import pandas as pd
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'completed.json').exists():
        print('Completed loss study exists; see report.md. No training or test evaluation repeated.');return
    jobs=json.loads((ROOT/'outputs/experiments/normalization_multiseed/jobs.json').read_text())
    protected=[ROOT/'checkpoints/best.pt',ROOT/'checkpoints/inference.json']
    protected += [p for p in (ROOT/'outputs').iterdir() if p.is_file() and p.suffix in {'.json','.csv'}]
    protocol_path=STUDY/'protocol.json'
    proposal=dict(loss='Tversky(alpha=0.3,beta=0.7)+BCE',epochs=30,seeds=[42,43,44],thresholds=THRESHOLDS,
                  reference_threshold=.70,validation_patients=json.loads((ROOT/'outputs/split.json').read_text())['val'],
                  test_patients_used=0,reference_checkpoints={str(seed):jobs[f'whole_volume_seed{seed}']['checkpoint_sha256'] for seed in [42,43,44]},
                  gates=dict(min_recall_gain=.01,max_dice_loss=.002,max_precision_loss=.02,max_false_alarm_increase=0,
                             max_patient_dice_loss=.05,missed_slices_must_decrease=True),
                  source_sha256={name:fingerprint(ROOT/name) for name in ['src/train.py','src/config.py','src/objectives.py']},
                  protected_sha256={str(p):fingerprint(p) for p in protected})
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text());assert protocol==proposal,'Protocol changed'
    else:
        protocol=proposal;save(protocol_path,protocol)
    tables=[];screen=None
    for seed in [42,43,44]:
        if seed!=42 and not screen['passes']:break
        output=STUDY/f'seed{seed}';output.mkdir(exist_ok=True)
        completion=output/'completed.json'
        if not completion.exists():
            assert not (output/'history.csv').exists(),'Partial training retained; investigate before a fresh attempt'
            save(STUDY/'status.json',dict(stage='training',seed=seed,epochs=30))
            print(f'Starting candidate seed {seed}',flush=True)
            with (output/'training.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-u',__file__,'--worker','--seed',str(seed)],cwd=ROOT,
                                      stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if result.returncode:raise RuntimeError(f'Training failed for seed {seed}; inspect {output}/training.log')
        completed=json.loads(completion.read_text())
        assert fingerprint(CHECKPOINTS/f'seed{seed}/best.pt')==completed['checkpoint_sha256']
        save(STUDY/'status.json',dict(stage='validation',seed=seed))
        tables.append(measure(seed,protocol))
        if seed==42:
            screen=screening(tables[0]);save(STUDY/'screening.json',screen)
            print(f"Seed-42 prespecified screen passed: {screen['passes']}",flush=True)
    report(protocol,tables,screen)
    save(STUDY/'status.json',dict(stage='complete',completed_seeds=len(tables),screen_passed=screen['passes']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--study',choices=['recall_loss','recall_cleanup'],default='recall_loss')
    args=parser.parse_args()
    if args.study!='recall_loss':
        if not args.worker:parser.error('Follow-up study paths are available only to the training worker')
        STUDY=ROOT/'outputs/experiments'/args.study
        CHECKPOINTS=ROOT/'checkpoints/experiments'/args.study
    if args.worker:worker(args.seed)
    else:run()
