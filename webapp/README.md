# Web interface (in progress)

Planned flow:

1. **Enroll:** user records 3–5 short clips → preprocessed → ECAPA-TDNN embeddings averaged and stored as the voiceprint.
2. **Verify:** user records one clip → AASIST checks it is bona fide → ECAPA-TDNN compares it to the stored voiceprint → access granted only if both pass.
