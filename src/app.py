"""Single-slice Gradio inference with patient-scan normalization metadata."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import find_boundaries
from .config import Config
from .data import normalize, read_image
from .model import load_model
from .utils import device

def overlay(image: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    display = image.astype(np.float32)
    display = (255 * (display - display.min()) / max(float(np.ptp(display)), 1)).astype(np.uint8)
    display[find_boundaries(prediction, mode="inner")] = [255, 0, 0]
    return display

class Predictor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.dev = device()
        self.model, self.checkpoint = load_model(config, self.dev)

    @torch.no_grad()
    def predict(self, path: str, patient: str) -> tuple[np.ndarray, int]:
        image = read_image(path) if Path(path).suffix.lower() in (".tif", ".tiff") else np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        stats = self.checkpoint["normalization"].get(patient)
        if stats is None:
            raise ValueError("Select the matching patient normalization profile. For a new scan, compute statistics across the complete patient scan before inference.")
        tensor = torch.from_numpy(normalize(image, stats).transpose(2, 0, 1).copy())[None].to(self.dev)
        height, width = image.shape[:2]
        tensor = torch.nn.functional.pad(tensor, (0, (-width) % 32, 0, (-height) % 32))
        prediction = (self.model(tensor).sigmoid()[0, 0, :height, :width] >= self.config.prediction_threshold).cpu().numpy()
        return overlay(image, prediction), int(prediction.sum())

def save_sample(config: Config, image: str, patient: str) -> None:
    result, _ = Predictor(config).predict(image, patient)
    Image.fromarray(result).save(config.output_dir / "demo_overlay.png")

def launch() -> None:
    import gradio as gr
    predictor = Predictor(Config())
    profiles = sorted(predictor.checkpoint["normalization"])
    threshold = predictor.config.prediction_threshold
    print(f"Inference probability threshold: {threshold:.2f}", flush=True)
    model_id = predictor.checkpoint["checkpoint_sha256"]
    epoch = predictor.checkpoint["epoch"]
    print(f"Checkpoint SHA-256: {model_id}, epoch {epoch}", flush=True)
    gr.Interface(
        fn=predictor.predict,
        inputs=[
            gr.File(type="filepath", label="MRI slice", file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"]),
            gr.Dropdown(profiles, label="Matching patient normalization profile"),
        ],
        outputs=[gr.Image(label="Predicted tumor outline"), gr.Number(label="Tumor area (pixels)")],
        title="LGG MRI tumor segmentation",
        description=(
            "Upload a slice with channels ordered pre-contrast, FLAIR, post-contrast. "
            "PNG/JPG channel values must preserve the MRI channel representation used for training. "
            f"Select its patient scan profile. Probability threshold {threshold:.2f}. "
            f"Model ID {model_id[:12]}, epoch {epoch}."
        ),
    ).launch()

if __name__ == "__main__":
    launch()
