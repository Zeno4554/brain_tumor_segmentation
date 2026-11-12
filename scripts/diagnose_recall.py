"""Describe validation failure sizes using cached probabilities; no test access."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.baseline_tuned import patient_files
from src.config import Config
from src.data import read_image,read_mask
from src.utils import segmentation_metrics


def main():
    destination=ROOT/'outputs/experiments/recall_loss'
    destination.mkdir(parents=True,exist_ok=True)
    split=json.loads((ROOT/'outputs/split.json').read_text())
    frame=patient_files(Config(),split['val'])
    rows=[]
    for patient,group in frame.groupby('patient'):
        group=group.sort_values('index')
        truth=np.stack([read_mask(r.mask) for r in group.itertuples()]).astype(bool)
        for name in ['shipped','seed42']:
            path=ROOT/f'outputs/experiments/recall_refinement/{name}/probabilities/{patient}.npz'
            prob=np.load(path)['probabilities']
            assert prob.shape==truth.shape
            for position,r in enumerate(group.itertuples()):
                area=int(truth[position].sum())
                size='tumor_free' if area==0 else '1-99 pixels' if area<100 else '100-499 pixels' if area<500 else '>=500 pixels'
                rows.append(dict(model=name,patient=patient,index=r.index,position=position,
                                 size_group=size,**segmentation_metrics(prob[position]>=.70,truth[position]),
                                 maximum_probability=float(prob[position].max()),
                                 maximum_probability_on_annotation=float(prob[position][truth[position]].max()) if area else None))
    table=pd.DataFrame(rows)
    table.to_csv(destination/'development_slice_diagnostics.csv',index=False)
    summary=table.groupby(['model','size_group']).agg(slices=('patient','size'),
        completely_missed=('completely_missed_positive_slices','sum'),zero_overlap=('zero_overlap_positive_slices','sum'),
        annotated_pixels=('annotated_pixels','sum'),missed_pixels=('false_negative_pixels','sum'),
        false_alarms=('false_alarms','sum')).reset_index()
    summary.to_csv(destination/'development_error_sizes.csv',index=False)
    missed=table[table.completely_missed_positive_slices>0].sort_values(['model','annotated_pixels'],ascending=[True,False])
    missed.to_csv(destination/'development_fully_missed_slices.csv',index=False)
    from matplotlib.figure import Figure
    selected=missed[missed.model=='seed42'].head(3)
    fig=Figure(figsize=(10,10));axes=fig.subplots(3,2)
    intensity_rows=[]
    for i,r in enumerate(selected.itertuples()):
        folder=Config().data_dir/r.patient
        image=read_image(folder/f'{r.patient}_{r.index}.tif')
        annotation=read_mask(folder/f'{r.patient}_{r.index}_mask.tif').astype(bool)
        probabilities=np.load(ROOT/f'outputs/experiments/recall_refinement/seed42/probabilities/{r.patient}.npz')['probabilities'][r.position]
        flair=image[...,1]
        intensity_rows.append(dict(patient=r.patient,index=r.index,flair_max=float(flair.max()),
            flair_std=float(flair.std()),flair_mean_on_annotation=float(flair[annotation].mean()),
            flair_mean_nonzero=float(flair[flair>0].mean()),annotated_pixels=int(annotation.sum()),
            max_probability=float(probabilities.max())))
        axes[i,0].imshow(flair,cmap='gray');axes[i,0].contour(annotation,levels=[.5],colors=['lime'],linewidths=1)
        axes[i,0].set_title(f'{r.patient}, slice {r.index}',fontsize=9)
        im=axes[i,1].imshow(probabilities,cmap='magma',vmin=0,vmax=1)
        axes[i,1].contour(annotation,levels=[.5],colors=['lime'],linewidths=1)
        axes[i,1].set_title(f'Probability; maximum {probabilities.max():.4f}')
        for ax in axes[i]:ax.axis('off')
    fig.colorbar(im,ax=list(axes[:,1]),fraction=.025,pad=.01)
    fig.savefig(destination/'validation_missed_examples.png',dpi=140)
    pd.DataFrame(intensity_rows).to_csv(destination/'validation_missed_intensities.csv',index=False)
    print(summary.to_string(index=False))
    print('\nLargest completely missed validation slices:\n'+missed[['model','patient','index','annotated_pixels','maximum_probability']].head(8).to_string(index=False))


if __name__=='__main__':main()
