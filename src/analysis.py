"""Small reusable notebook visualizations and failure analysis."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import Config
from .data import prepare, read_image, read_mask
from .evaluate import predict_patient
from .model import load_model
from .utils import device, dice

def explore(config: Config | None = None) -> None:
    frame, _, _ = prepare(config or Config())
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for axes_row, (_, group) in zip(axes, list(frame.groupby("patient"))[:3]):
        row = group.iloc[len(group) // 2]
        image = read_image(row.image)
        axes_row[0].imshow(image / max(float(image.max()), 1))
        axes_row[1].imshow(read_mask(row['mask']), cmap="gray")
        axes_row[2].imshow(image[..., 1], cmap="gray")
        for ax, title in zip(axes_row, [row.patient, "Ground truth", "FLAIR"]):
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    plt.show()

def failures(config: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or Config()
    scores = pd.read_csv(config.output_dir / "per_patient_dice.csv").sort_values("unet_dice")
    dev = device()
    model, checkpoint = load_model(config, dev)
    frame, splits, stats = prepare(config)
    assert checkpoint["splits"] == splits
    records = []
    worst = set(scores.head(5).patient)
    for patient in scores.patient:
        group = frame[frame.patient == patient].sort_values("index")
        truth = np.stack([read_mask(r.mask) for r in group.itertuples()])
        pred = predict_patient(model, group, stats, config, dev)
        slice_scores = [dice(p, t) for p, t in zip(pred, truth)]
        if patient in worst:
            for i in np.argsort(slice_scores)[:3]:
                fig, axes = plt.subplots(1, 3, figsize=(10, 3))
                image = read_image(group.iloc[i].image)
                for ax, content, title in zip(axes, [image[..., 1], truth[i], pred[i]], [f"{patient} slice {group.iloc[i]['index']}", "Ground truth", f"Prediction, Dice {slice_scores[i]:.3f}"]):
                    ax.imshow(content, cmap="gray")
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                fig.tight_layout()
                plt.show()
        size = int(truth.sum())
        size_group = "small" if size < 10000 else "medium" if size < 50000 else "large"
        records.append(dict(patient=patient, size=size_group, region="whole scan", dice=dice(pred, truth)))
        edge = max(1, int(np.ceil(len(group) * 0.2)))
        for region, selection in [("bottom 20%", slice(0, edge)), ("middle 60%", slice(edge, -edge)), ("top 20%", slice(-edge, None))]:
            if len(truth[selection]):
                records.append(dict(patient=patient, size=size_group, region=region, dice=dice(pred[selection], truth[selection])))
    detail = pd.DataFrame(records)
    sizes = detail[detail.region == "whole scan"].groupby("size").dice.agg(["count", "mean", "median"])
    regions = detail[detail.region != "whole scan"].groupby("region").dice.agg(["count", "mean", "median"])
    sizes.to_csv(config.output_dir / "failure_by_size.csv")
    regions.to_csv(config.output_dir / "failure_by_region.csv")
    print("Tumor size uses total annotated voxels per patient: small <10,000, medium <50,000, large >=50,000.")
    print(sizes.to_string())
    print("Regions follow slice index order. Confirm anatomical direction against scan metadata.")
    print(regions.to_string())
    return sizes, regions
