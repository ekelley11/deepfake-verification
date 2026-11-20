"""Inference test for speaker verification (ECAPA-TDNN, SpeechBrain).

Usage:
    python verification/test_verification.py <audio_a> <audio_b>

Prints a cosine-similarity score and whether the two clips are judged
to be the same speaker. Run it on two clips of the same person and on
two clips of different people, and screenshot both outputs.
"""
import sys
from pathlib import Path

from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    a, b = sys.argv[1], sys.argv[2]

    # COPY instead of symlink: symlinks fail on Windows without Developer Mode.
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODEL_DIR),
        local_strategy=LocalStrategy.COPY,
    )

    score, same = model.verify_files(a, b)
    print("Model:      ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)")
    print(f"Clip A:     {a}")
    print(f"Clip B:     {b}")
    print(f"Score:      {score.item():.4f}  (cosine similarity, higher = more similar)")
    print(f"Same speaker: {bool(same.item())}")


if __name__ == "__main__":
    main()
