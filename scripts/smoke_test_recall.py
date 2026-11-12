"""Synthetic objective/refinement checks and one isolated training epoch."""
import os
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
import segmentation_models_pytorch as smp
from src.objectives import SegmentationObjective
from src.config import Config
from scripts.smoke_test import synthetic_data
from scripts.recall_refinement_experiment import smoke


def main():
    smoke()
    import numpy as np
    from src.refinement import remove_small_regions
    mask=np.zeros((2,8,8),bool);mask[:,1:4,1:4]=True;mask[0,7,7]=True
    cleaned=remove_small_regions(mask,5)
    assert cleaned.sum()==18 and np.all(cleaned<=mask)
    assert np.array_equal(remove_small_regions(mask,0),mask)
    x=torch.zeros((1,1,2,2),requires_grad=True)
    y=torch.tensor([[[[1.,1.],[0.,0.]]]])
    default=SegmentationObjective()(x,y)
    original=smp.losses.DiceLoss('binary')(x,y)+torch.nn.BCEWithLogitsLoss()(x,y)
    torch.testing.assert_close(default,original,rtol=0,atol=0)
    reference_gradient=torch.autograd.grad(default,x,retain_graph=True)[0]
    candidate=SegmentationObjective('tversky_bce')(x,y)
    gradient=torch.autograd.grad(candidate,x)[0]
    assert torch.isfinite(gradient).all()
    assert abs(gradient[0,0,0,0])>abs(reference_gradient[0,0,0,0])
    assert abs(gradient[0,0,1,0])<abs(reference_gradient[0,0,1,0])
    from src.train import train
    with tempfile.TemporaryDirectory(prefix='recall-loss-smoke-') as directory:
        root=Path(directory)
        synthetic_data(root/'data')
        config=Config(data_dir=root/'data',output_dir=root/'outputs',checkpoint_dir=root/'checkpoints',
                      smoke_test=True,epochs=1,encoder_weights=None,loss_name='tversky_bce')
        checkpoint=train(config)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['loss_name']=='tversky_bce' and saved['epoch']==1
        assert saved['smoke_test'] and len(saved['splits']['test'])==2
    print('RECALL TRAINING/REFINEMENT SYNTHETIC SMOKE PASSED')


if __name__=='__main__':
    main()
