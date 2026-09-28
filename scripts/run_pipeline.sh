#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end order. Replace placeholder audio paths first.
python scripts/check_dataset.py
python scripts/evaluate_stt.py --model openai/whisper-small --manifest data/stt/test.jsonl --output reports/stt_before.json
python src/stt_train.py --dataset data/stt/corrections.jsonl --base-model openai/whisper-small --output models/stt/v1
python scripts/evaluate_stt.py --model models/stt/v1 --manifest data/stt/test.jsonl --output reports/stt_v1.json
python src/tts_train.py --dataset data/tts/corrections.jsonl --model facebook/mms-tts-tam --output models/tts/tam/v1
python scripts/evaluate_tts.py --model facebook/mms-tts-tam --manifest data/tts/test.jsonl --output-dir outputs/tts_before --report reports/tts_before.json
python scripts/evaluate_tts.py --model models/tts/tam/v1 --manifest data/tts/test.jsonl --output-dir outputs/tts_v1 --report reports/tts_v1.json
