"""Prespecified follow-up: isolate small false positives in the recall-loss candidate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1')
STUDY=ROOT/'outputs/experiments/recall_cleanup'
AREAS=[0,10,20,50,100,200]
THRESHOLDS=[round(.30+.05*i,2) for i in range(11)]


def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    from scripts.recall_loss_experiment import save as atomic_save
    atomic_save(path,value)


def measure(seed,protocol):
    import numpy as np
    import pandas as pd
    import torch
    from scipy import ndimage as ndi
    from src.config import Config
    from src.baseline_tuned import patient_files
    from src.data import read_mask
    from src.evaluate import predict_probabilities
    from src.model import load_model
    from src.utils import device,seed_everything,segmentation_metrics
    csv=STUDY/f'seed{seed}_patient_metrics.csv'
    if csv.exists():return pd.read_csv(csv)
    seed_everything(seed)
    if seed==42:
        checkpoint=ROOT/'checkpoints/experiments/recall_loss/seed42'
        cache=ROOT/'outputs/experiments/recall_loss/seed42/probabilities/candidate'
    else:
        checkpoint=ROOT/f'checkpoints/experiments/recall_cleanup/seed{seed}'
        cache=STUDY/f'seed{seed}/probabilities';cache.mkdir(parents=True,exist_ok=True)
    config=Config(checkpoint_dir=checkpoint,output_dir=STUDY/f'seed{seed}',prediction_threshold=.7)
    frame=patient_files(config,protocol['validation_patients'])
    model=None;dev=None;rows=[]
    for patient,group in frame.groupby('patient'):
        group=group.sort_values('index')
        truth=np.stack([read_mask(r.mask) for r in group.itertuples()]).astype(bool)
        cached=cache/f'{patient}.npz'
        if cached.exists():prob=np.load(cached)['probabilities']
        else:
            if model is None:
                dev=device();model,ckpt=load_model(config,dev)
                assert ckpt['loss_name']=='tversky_bce' and ckpt['seed']==seed
                assert ckpt['splits']['val']==protocol['validation_patients']
            prob=predict_probabilities(model,group,ckpt['normalization'],config,dev)
            np.savez_compressed(cached,probabilities=prob)
        for threshold in THRESHOLDS:
            labels=[ndi.label(image>=threshold,np.ones((3,3),bool))[0] for image in prob]
            sizes=[np.bincount(label.ravel()) for label in labels]
            for area in AREAS:
                slices=[]
                for label,counts in zip(labels,sizes):
                    keep=counts>=area;keep[0]=False;slices.append(keep[label])
                pred=np.stack(slices)
                rows.append(dict(seed=seed,patient=patient,threshold=threshold,min_area=area,
                                 **segmentation_metrics(pred,truth)))
    if model is not None:
        del model
        if dev.type=='cuda':torch.cuda.empty_cache()
    table=pd.DataFrame(rows);table.to_csv(csv,index=False)
    return table


def reference(seed):
    import pandas as pd
    path=ROOT/'outputs/experiments/recall_refinement'/f'seed{seed}_patient_metrics.csv'
    data=pd.read_csv(path)
    return data[data.low==.7]


def assess(table,threshold,area):
    import numpy as np
    from scripts.recall_refinement_experiment import aggregate
    results=[]
    for seed,group in table.groupby('seed'):
        ref=reference(int(seed));base=aggregate(ref)
        chosen=group[(group.threshold==threshold)&(group.min_area==area)]
        measured=aggregate(chosen)
        paired=ref[['patient','dice']].merge(chosen[['patient','dice']],on='patient',suffixes=('_reference','_candidate'))
        assert len(paired)==14
        changes={key:measured[key]-base[key] for key in base}
        results.append(dict(seed=int(seed),reference=base,candidate=measured,changes=changes,
                            worst_patient_dice_change=float((paired.dice_candidate-paired.dice_reference).min())))
    mean={key:float(np.mean([r['changes'][key] for r in results])) for key in results[0]['changes']}
    passes=(mean['pooled_recall']>=.01 and mean['mean_dice']>=-.002 and mean['missed_slices']<0
            and all(r['changes']['pooled_recall']>=0 and r['changes']['mean_dice']>=-.005
                    and r['changes']['pooled_precision']>=-.02 and r['changes']['false_alarms']<=0
                    and r['changes']['missed_slices']<=0 and r['worst_patient_dice_change']>=-.05 for r in results))
    return dict(threshold=threshold,min_area=area,passes=passes,mean_changes=mean,comparisons=results)


def run():
    import pandas as pd
    from scripts.recall_refinement_experiment import aggregate
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'completed.json').exists():print('Completed follow-up exists; no repeated training or test evaluation.');return
    prior=json.loads((ROOT/'outputs/experiments/recall_loss/completed.json').read_text())
    assert not prior['screen_passed'], 'This follow-up documents the rejected seed-42 screen'
    original_protocol=json.loads((ROOT/'outputs/experiments/recall_loss/protocol.json').read_text())
    proposal=dict(thresholds=THRESHOLDS,min_areas=AREAS,validation_patients=original_protocol['validation_patients'],
                  reused_seed42_checkpoint_sha256=fingerprint(ROOT/'checkpoints/experiments/recall_loss/seed42/best.pt'),
                  reference_checkpoints=original_protocol['reference_checkpoints'],
                  reason='Seed-42 loss trial improved recall and missed slices but increased false positives. This separate development follow-up tests small-component cleanup; it does not change the original loss-study rejection.',
                  gates=dict(mean_recall_gain_at_least=.01,mean_dice_loss_at_most=.002,each_seed_dice_loss_at_most=.005,
                             each_seed_precision_loss_at_most=.02,each_seed_recall_must_not_fall=True,
                             no_more_false_alarm_slices_in_any_seed=True,no_more_missed_slices_in_any_seed=True,
                             average_missed_slices_must_decrease=True,max_patient_dice_loss=.05),
                  selection='Screen seed 42; among passing settings maximize mean Dice, then recall, smaller area, threshold nearest 0.7. Freeze before training seeds 43/44; evaluate the same setting on them.',
                  source_sha256={p:fingerprint(ROOT/p) for p in ['src/train.py','src/config.py','src/objectives.py']},
                  protected_sha256=original_protocol['protected_sha256'],test_patients_used=0)
    protocol_path=STUDY/'protocol.json'
    if protocol_path.exists():protocol=json.loads(protocol_path.read_text());assert protocol==proposal
    else:protocol=proposal;save(protocol_path,protocol)
    first=measure(42,protocol)
    choices=[assess(first,t,a) for t in THRESHOLDS for a in AREAS]
    save(STUDY/'screening.json',dict(choices=choices))
    eligible=[c for c in choices if c['passes']]
    chosen=max(eligible,key=lambda c:(c['comparisons'][0]['candidate']['mean_dice'],c['comparisons'][0]['candidate']['pooled_recall'],-c['min_area'],-abs(c['threshold']-.7))) if eligible else None
    selection=dict(selected=chosen,screen_passed=bool(eligible))
    selection_path=STUDY/'frozen_selection.json'
    if selection_path.exists():assert json.loads(selection_path.read_text())==selection
    else:save(selection_path,selection)
    print('Cleanup screen passed:',bool(eligible),flush=True)
    tables=[first]
    if chosen:
        print('FROZEN candidate cleanup:',chosen['threshold'],chosen['min_area'],flush=True)
        for seed in [43,44]:
            output=STUDY/f'seed{seed}';output.mkdir(exist_ok=True)
            if not (output/'completed.json').exists():
                assert not (output/'history.csv').exists(),'Preserving partial training; inspect before retry'
                save(STUDY/'status.json',dict(stage='training',seed=seed))
                with (output/'training.log').open('w',encoding='utf-8') as log:
                    result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/recall_loss_experiment.py'),'--worker','--seed',str(seed),'--study','recall_cleanup'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if result.returncode:raise RuntimeError(f'Follow-up seed {seed} failed; inspect training.log')
            record=json.loads((output/'completed.json').read_text())
            assert fingerprint(ROOT/f'checkpoints/experiments/recall_cleanup/seed{seed}/best.pt')==record['checkpoint_sha256']
            save(STUDY/'status.json',dict(stage='validation',seed=seed))
            tables.append(measure(seed,protocol))
    table=pd.concat(tables,ignore_index=True);table.to_csv(STUDY/'validation_patient_metrics.csv',index=False)
    summary=pd.DataFrame([dict(seed=int(seed),threshold=float(t),min_area=int(a),**aggregate(g)) for (seed,t,a),g in table.groupby(['seed','threshold','min_area'])])
    summary.to_csv(STUDY/'validation_summary.csv',index=False)
    final=assess(table,chosen['threshold'],chosen['min_area']) if chosen else None
    accepted=bool(final and final['passes'] and len(tables)==3)
    decision=dict(screen_passed=bool(chosen),confirmed_across_three_seeds=accepted,seeds_evaluated=len(tables),
                  final=final,shipped_model_changed=False,test_patients_evaluated=0,
                  decision='Promising development candidate; independent confirmation required' if accepted else 'No qualifying follow-up; keep shipped model')
    lines=['# Small-component follow-up','',protocol['reason'],'',
           'A fresh protocol freezes threshold/area grids and acceptance gates before cleanup measurements. Seed 42 screens settings on the reused 14 validation patients; any passing setting is frozen before training matched seeds 43/44. This is additional development selection on a small reused cohort, not independent validation. No test images or masks were used.','',
           f"**{decision['decision']}.** Seed-42 screen passed: {bool(chosen)}; evaluated seeds: {len(tables)}.",'',
           '| Seed | Model | Threshold | Min area | Dice | Precision | Recall | False alarms | Missed slices |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    if final:
        for c in final['comparisons']:
            for role in ['reference','candidate']:
                m=c[role]
                lines.append(f"| {c['seed']} | {role} | {(.7 if role=='reference' else final['threshold']):.2f} | {(0 if role=='reference' else final['min_area'])} | {m['mean_dice']:.4f} | {m['pooled_precision']:.2%} | {m['pooled_recall']:.2%} | {m['false_alarms']}/269 | {m['missed_slices']} |")
    else:
        best=summary.sort_values('mean_dice',ascending=False).iloc[0]
        base=aggregate(reference(42))
        for role,threshold,area,m in [('reference',.70,0,base),('highest Dice; rejected',float(best.threshold),int(best.min_area),best.to_dict())]:
            lines.append(f"| 42 | {role} | {threshold:.2f} | {area} | {m['mean_dice']:.4f} | {m['pooled_precision']:.2%} | {m['pooled_recall']:.2%} | {int(m['false_alarms'])}/269 | {int(m['missed_slices'])} |")
        lines += ['',f"Highest seed-42 Dice among the fixed cleanup grid: {best.mean_dice:.4f}, threshold {best.threshold:.2f}, area {int(best.min_area)}, precision {best.pooled_precision:.2%}, recall {best.pooled_recall:.2%}, false alarms {int(best.false_alarms)}/269, missed slices {int(best.missed_slices)}. No setting met every gate; this descriptive row is not an accepted selection."]
    lines += ['', 'No deployment was changed. Removing components can erase real small tumors; per-patient Dice, recall and missed-slice gates explicitly check that tradeoff. The loss-only trial remains recorded as a rejected primary screen. New patients and expert review remain required.','']
    for p,h in protocol['protected_sha256'].items():assert fingerprint(p)==h,p
    (STUDY/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    save(STUDY/'completed.json',dict(**decision,protected_artifacts_unchanged=True))
    save(STUDY/'status.json',dict(stage='complete',confirmed=accepted))
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':run()
