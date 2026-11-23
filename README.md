# Deepfake-Resistant Voice Verification

A voice-based identity system that answers two questions for every login attempt:

1. **Verification:** is this the enrolled speaker? (ECAPA-TDNN via SpeechBrain)
2. **Spoof detection:** is this real human speech or AI-generated / replayed? (AASIST)

Access is granted only if the speaker matches **and** the audio is judged bona fide.

## Repository layout

```
verification/    Speaker verification (ECAPA-TDNN) inference test
detection/       Spoof / deepfake detection (AASIST) inference test
preprocessing/   Audio preprocessing: mono, 16 kHz resample, silence trim, normalize
webapp/          Enrollment + verification web interface (in progress)
docs/            Consent form template, report material
data/            Local audio only. Never committed (see .gitignore)
```

## Setup (Windows, Python 3.11)

Run these in a terminal opened in the repo folder:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Always run scripts with `.venv\Scripts\python` (as below). Plain `pip`/`python` can
silently use your system Python instead of the project environment.

Then fetch the AASIST code and pretrained weights (one time):

```bash
git clone https://github.com/clovaai/aasist.git detection/aasist
```

## Inference tests

```bash
.venv\Scripts\python verification/test_verification.py data/samples/a.wav data/samples/b.wav
.venv\Scripts\python detection/test_detection.py data/samples/a.wav
.venv\Scripts\python preprocessing/preprocess.py data/raw data/processed
```

## Models

| Role | Model | Citation |
|---|---|---|
| Speaker verification | ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) | Desplanques, Thienpondt, Demuynck. *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.* Interspeech 2020. |
| Toolkit | SpeechBrain | Ravanelli et al. *SpeechBrain: A General-Purpose Speech Toolkit.* arXiv:2106.04624, 2021. |
| Spoof detection | AASIST (pretrained on ASVspoof 2019 LA) | Jung et al. *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.* ICASSP 2022. |
| Benchmark data | ASVspoof 2019 Logical Access | Wang et al. *ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech.* Computer Speech & Language, 2020. |

## Labor division

| Member | Role | Owns |
|---|---|---|
| Elijah Kelley | PM / Integration | Repo, environment, README, final PDF |
| Lukas John | Verification | `verification/`, ECAPA-TDNN test + screenshot, citations |
| Wellborn Fuller| Detection | `detection/`, AASIST test + screenshot, model explanation |
| Roy Cannedy | Data & Preprocessing | `preprocessing/`, consent forms, enrollment recordings, benchmark samples |

## Data & consent

- Enrollment recordings are collected only after a signed consent form (`docs/consent_form_template.md`).
- Signed forms and all audio stay **out of this repo** (shared privately, e.g. Google Drive).
- Fake samples come from the public ASVspoof 2019 LA benchmark (small subset only).
