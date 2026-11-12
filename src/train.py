"""Mixed-precision patient-held-out U-Net training."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from .config import Config
from .data import MRIDataset, prepare, read_mask, seed_worker
from .model import build_model
from .objectives import SegmentationObjective
from .utils import device, seed_everything, write_json

def training_sampler(frame: pd.DataFrame, config: Config) -> WeightedRandomSampler | None:
    """Optionally target a positive-slice fraction using training masks only."""
    if config.target_positive_fraction is None:
        return None
    positive = np.asarray([read_mask(row.mask).any() for row in frame.itertuples()])
    positive_count = int(positive.sum())
    negative_count = len(positive) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("Targeted sampling requires both tumor-containing and tumor-free training slices")
    target = config.target_positive_fraction
    weights = np.where(positive, target / positive_count, (1 - target) / negative_count)
    write_json(config.output_dir / "sampling.json", dict(
        training_slices=len(frame), positive_slices=positive_count,
        negative_slices=negative_count, natural_positive_fraction=float(positive.mean()),
        target_positive_fraction=target, samples_per_epoch=len(frame), replacement=True,
    ))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(frame), replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

@torch.no_grad()
def validation_dice(model: torch.nn.Module, loader: DataLoader, dev: torch.device) -> float:
    model.eval()
    totals: dict[str, list[float]] = {}
    offset = 0
    for images, masks in loader:
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            pred = model(images.to(dev)).sigmoid() >= 0.5
        truth = masks.to(dev).bool()
        for i in range(len(images)):
            patient = loader.dataset.frame.iloc[offset + i].patient
            counts = totals.setdefault(patient, [0., 0., 0.])
            counts[0] += (pred[i] & truth[i]).sum().item()
            counts[1] += pred[i].sum().item()
            counts[2] += truth[i].sum().item()
        offset += len(images)
    return float(np.mean([(2 * a + 1e-7) / (b + c + 1e-7) for a, b, c in totals.values()]))

def train(config: Config | None = None) -> Path:
    config = config or Config()
    seed_everything(config.seed)
    dev = device()
    frame, splits, stats = prepare(config)
    generator = torch.Generator().manual_seed(config.seed)
    training_frame = frame[frame.patient.isin(splits["train"])]
    sampler = training_sampler(training_frame, config)
    train_loader = DataLoader(
        MRIDataset(training_frame, stats, True, config.seed),
        batch_size=config.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=config.num_workers, generator=generator,
        pin_memory=dev.type == "cuda", worker_init_fn=seed_worker,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        MRIDataset(frame[frame.patient.isin(splits["val"])], stats),
        batch_size=config.batch_size, num_workers=0,
        generator=torch.Generator().manual_seed(config.seed + 1),
    )
    try:
        model = build_model(config).to(dev)
        optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 1e-4}, {"params": list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), "lr": 1e-3}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        objective = SegmentationObjective(config.loss_name)
        best, history = -1., []
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(config.epochs):
            model.train()
            total_loss = 0.
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}"):
                images, masks = images.to(dev), masks.to(dev)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    logits = model(images)
                    loss = objective(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(images)
            score = validation_dice(model, val_loader, dev)
            scheduler.step(score)
            history.append(dict(epoch=epoch + 1, train_loss=total_loss / len(train_loader.dataset), val_dice=score, encoder_lr=optimizer.param_groups[0]["lr"], decoder_lr=optimizer.param_groups[1]["lr"]))
            pd.DataFrame(history).to_csv(config.output_dir / "history.csv", index=False)
            print(f"Validation volume Dice: {score:.4f}")
            if score > best:
                best = score
                temporary = config.checkpoint_dir / "best.tmp"
                torch.save(dict(model=model.state_dict(), encoder=config.encoder, epoch=epoch + 1, val_dice=score, splits=splits, normalization=stats, smoke_test=config.smoke_test, target_positive_fraction=config.target_positive_fraction, normalization_mode=config.normalization_mode, seed=config.seed, augmentation_seeding="worker_specific_persistent", loss_name=config.loss_name), temporary)
                os.replace(temporary, config.checkpoint_dir / "best.pt")
            if epoch == 0 and dev.type == "cuda":
                print(f"Peak epoch-1 GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    except torch.cuda.OutOfMemoryError:
        print('GPU out of memory. Lower BATCH_SIZE in src/config.py or set ENCODER="resnet18".')
        raise
    return config.checkpoint_dir / "best.pt"

if __name__ == "__main__":
    train()
