# Veltech.AI – STT & TTS Model Training Technical Exercise

This repository is the implementation base for the supplied exercise requirements.

## Required final flow

```text
Base Model
  → Incorrect Output
  → Human Correction
  → Training Data
  → Fine-Tuning
  → Updated Model
  → Retest
  → Improved Output
```

## What is included
- Whisper STT inference and fine-tuning
- MMS/VITS TTS inference and model-level fine-tuning
- Streamlit human correction collection
- Persistent JSONL correction datasets
- Held-out STT WER evaluation
- Held-out TTS audio generation + human-review report
- Versioned model output structure
- Continued-learning procedure
- Technical documentation and requirement checklist

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU systems, install the PyTorch build appropriate for the installed CUDA version.

## 2. Run correction UI

```bash
streamlit run app.py
```

The UI only stores human corrections. It does not hardcode replacements into inference.

## 3. Record/collect data

Put real speech recordings in:

```text
data/stt/audio/
data/tts/audio/
```

Update the JSONL manifests with the real paths. Keep a held-out test set separate from training corrections.

## 4. Test base STT

```bash
python src/stt_infer.py --audio data/stt/audio/your_test.wav --model openai/whisper-small --language ta
```

## 5. Test base TTS

```bash
python src/tts_infer.py --text "வணக்கம் அருண்" --language tam --output outputs/tts_before.wav
```

MMS checkpoints are available per supported language; verify the chosen checkpoint and its tokenizer requirements before claiming language coverage.

## 6. Add corrections

STT:

```bash
python scripts/add_correction.py stt --audio data/stt/audio/correction.wav --language ta --model-output "wrong output" --corrected-text "correct output"
```

TTS:

```bash
python scripts/add_correction.py tts --text "அருண்" --language tam --reference-audio data/tts/audio/arun_reference.wav
```

## 7. Train STT v1

```bash
python src/stt_train.py --dataset data/stt/corrections.jsonl --base-model openai/whisper-small --output models/stt/v1
```

## 8. Evaluate STT before/after

```bash
python scripts/evaluate_stt.py --model openai/whisper-small --manifest data/stt/test.jsonl --output reports/stt_before.json
python scripts/evaluate_stt.py --model models/stt/v1 --manifest data/stt/test.jsonl --output reports/stt_v1.json
```

Compare measured WER. Never invent results.

## 9. Train TTS v1

```bash
python src/tts_train.py --dataset data/tts/corrections.jsonl --model facebook/mms-tts-tam --output models/tts/tam/v1
```

Use enough genuine reference recordings for a meaningful experiment. The supplied script is a compact VITS/MMS spectrogram-loss fine-tuning recipe, not a claim of production-scale TTS training.

## 10. Evaluate TTS

```bash
python scripts/evaluate_tts.py --model facebook/mms-tts-tam --manifest data/tts/test.jsonl --output-dir outputs/tts_before --report reports/tts_before.json
python scripts/evaluate_tts.py --model models/tts/tam/v1 --manifest data/tts/test.jsonl --output-dir outputs/tts_v1 --report reports/tts_v1.json
```

Perform human listening review on the same held-out texts.

## 11. Continued learning v2

Add new corrections, retain all old corrections, then train from the previous version or base checkpoint using the complete retained dataset:

```bash
python src/stt_train.py --dataset data/stt/corrections.jsonl --base-model models/stt/v1 --output models/stt/v2
python src/tts_train.py --dataset data/tts/corrections.jsonl --model models/tts/tam/v1 --output models/tts/tam/v2
```

Re-run the same held-out tests and record v1 vs v2 results.

## 12. Required final submission

- Working STT model
- Working TTS model
- Training/fine-tuning code
- Training and correction datasets
- Saved trained model/adapter and supporting files
- Technical documentation
- Before/after results
- Short continued-learning demonstration

## Important
This ZIP intentionally does not pretend to contain real speech recordings or measured improvement results. Those must be produced during the exercise using the candidate's own collected/recorded data. Placeholder audio paths in the manifests must be replaced before training.
