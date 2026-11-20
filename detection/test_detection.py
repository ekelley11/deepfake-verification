"""Inference test for spoof / deepfake detection (AASIST).

One-time setup (from the repo root):
    git clone https://github.com/clovaai/aasist.git detection/aasist

Usage:
    python detection/test_detection.py <audio> [<audio> ...]

Prints a bona fide score per clip. Higher = more likely real human speech,
lower = more likely spoofed / AI-generated. Run it on a few real and a few
fake ASVspoof clips and screenshot the output.

Scores are relative, not a fixed real/fake verdict: audio from outside
ASVspoof 2019 (different mics, accents, compression) shifts every score.
The accept/reject threshold gets calibrated on our own data later.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
AASIST_DIR = HERE / "aasist"
CONFIG = AASIST_DIR / "config" / "AASIST.conf"
WEIGHTS = AASIST_DIR / "models" / "weights" / "AASIST.pth"
NUM_SAMPLES = 64600  # ~4 s at 16 kHz, the input length AASIST was trained on


def load_model() -> torch.nn.Module:
    if not AASIST_DIR.exists():
        sys.exit("AASIST not found. Run: git clone https://github.com/clovaai/aasist.git detection/aasist")
    sys.path.insert(0, str(AASIST_DIR))
    from models.AASIST import Model  # noqa: E402  (import after path setup)

    with open(CONFIG) as f:
        model_config = json.load(f)["model_config"]
    model = Model(model_config)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_audio(path: str) -> torch.Tensor:
    wav, _ = librosa.load(path, sr=16000, mono=True)
    if len(wav) >= NUM_SAMPLES:
        wav = wav[:NUM_SAMPLES]
    else:  # repeat short clips to fill the window, as AASIST's own loader does
        wav = np.tile(wav, int(np.ceil(NUM_SAMPLES / len(wav))))[:NUM_SAMPLES]
    return torch.from_numpy(wav).float().unsqueeze(0)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    model = load_model()
    print("Model: AASIST (pretrained on ASVspoof 2019 LA)")
    print("Bona fide score: higher = more likely real speech, lower = more likely spoofed")
    with torch.no_grad():
        for path in sys.argv[1:]:
            _, logits = model(load_audio(path))
            score = logits[0, 1].item()  # index 1 = bona fide
            print(f"{Path(path).name:40s} score={score:8.4f}")


if __name__ == "__main__":
    main()
