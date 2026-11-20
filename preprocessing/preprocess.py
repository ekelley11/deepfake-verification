"""Audio preprocessing for both models.

Converts every audio file in <input_dir> to:
  - mono
  - 16 kHz sample rate (what ECAPA-TDNN and AASIST expect)
  - leading/trailing silence trimmed
  - peak-normalized
and writes .wav files to <output_dir>, keeping the subfolder structure.

Usage:
    python preprocessing/preprocess.py <input_dir> <output_dir>
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000
TOP_DB = 30  # how quiet (relative to peak) counts as silence
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def preprocess(path: Path) -> np.ndarray:
    wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=TOP_DB)
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = 0.95 * wav / peak
    return wav


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    files = [p for p in src.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        sys.exit(f"No audio files found in {src}")

    for p in files:
        out = dst / p.relative_to(src).with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        wav = preprocess(p)
        sf.write(out, wav, TARGET_SR)
        print(f"{p.name:40s} -> {out}  ({len(wav) / TARGET_SR:.2f} s)")
    print(f"Done: {len(files)} file(s).")


if __name__ == "__main__":
    main()
