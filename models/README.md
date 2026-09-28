# Model storage

Generated model versions are intentionally ignored by Git because weights can be large.

Recommended layout:

models/
  stt/v1/
  tts/tam/v1/
  tts/tam/v2/

Each version should contain model weights, processor/tokenizer, training manifest and metadata.
AI-2026 Veltech.AI