# Veltech.AI STT & TTS Model Training – Technical Documentation

## 1. Project Objective

This project implements a correction-driven training and continued-learning workflow for Speech-to-Text (STT) and Text-to-Speech (TTS).

Flow:

Base Model → Model Output → Human Correction → Correction Dataset → Fine-Tuning → Updated Model → Retest → Continued Learning

Human corrections become persistent training data. The project does not use hardcoded replacement rules as the learning mechanism.

## 2. Environment

- Python: 3.14.4
- PyTorch: 2.14.0+cpu
- Transformers: 5.17.0
- Hardware: CPU-only
- No NVIDIA GPU/CUDA GPU available

Main libraries include PyTorch, Transformers, Datasets, Accelerate, Evaluate, JiWER, Librosa, SoundFile, NumPy, Pandas, PyYAML, SentencePiece, Safetensors and Streamlit.

## 3. STT Model and Data

Base model: openai/whisper-small

Correction dataset:
data/stt/corrections.jsonl

Initial training examples:
1. வணக்கம் கவின்.
2. நான் இன்று அலுவலகத்திற்கு செல்கிறேன்.
3. வணக்கம், என் பெயர் கவின்.

Each correction record stores the audio path, language, model output, corrected text and correction source.

## 4. STT Fine-Tuning

Command:

python src/stt_train.py \
  --dataset data/stt/corrections.jsonl \
  --base-model openai/whisper-small \
  --output models/stt/v1 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 1

Result:
- Samples: 3
- Epochs: 1
- Learning rate: 1e-5
- Batch size: 1
- Gradient accumulation: 1
- Training loss: approximately 7.015
- Output: models/stt/v1

## 5. STT Evaluation

Held-out test samples:

1. Tamil: வணக்கம் கவின்
2. Numbers/date: Order 123 is ready on 26/09/2026.
3. Alphanumeric: AI-2026 Veltech.AI

Final measured result:
- Samples: 3
- WER: 1.2000

The fine-tuned model was successfully saved and reloadable. The held-out WER did not improve over the baseline. The dataset is very small, so no generalization improvement is claimed.

## 6. TTS Model and Data

Base model: facebook/mms-tts-tam

Correction dataset:
data/tts/corrections.jsonl

TTS correction records contain:
- Correct text
- Language
- Human reference audio
- Correction source

TTS requires human reference speech because text-only correction cannot teach the desired waveform pronunciation.

## 7. TTS Fine-Tuning Method

The installed Transformers VITS training interface does not support direct VITS training with the required labels.

Therefore, the project uses model-level spectrogram reconstruction fine-tuning:

1. Load pretrained MMS-TTS model.
2. Tokenize input text.
3. Generate model waveform.
4. Load human reference recording.
5. Convert generated and reference audio to spectrograms.
6. Calculate reconstruction loss.
7. Backpropagate through the model.
8. Update model parameters using AdamW.
9. Save the updated model.

This is model-level learning, not a pronunciation dictionary or post-processing replacement.

## 8. TTS v1

Command:

python src/tts_train.py \
  --dataset data/tts/corrections.jsonl \
  --model facebook/mms-tts-tam \
  --output models/tts/tam/v1 \
  --epochs 1 \
  --learning-rate 1e-6

Result:
- Samples: 3
- Epochs: 1
- Learning rate: 1e-6
- Optimizer: AdamW
- Average loss: approximately 0.529471
- Output: models/tts/tam/v1

## 9. Human TTS Corrections

After TTS v1 evaluation, two pronunciation corrections were collected:

data/tts/audio/TTS_CORRECTION_002.wav
data/tts/audio/TTS_CORRECTION_003.wav

The new recordings were added to the existing correction dataset.

## 10. Continued Learning / TTS v2

The complete retained correction dataset was used to train TTS v2 starting from TTS v1.

Command:

python src/tts_train.py \
  --dataset data/tts/corrections.jsonl \
  --model models/tts/tam/v1 \
  --output models/tts/tam/v2 \
  --epochs 1 \
  --learning-rate 1e-6

Result:
models/tts/tam/v2

Workflow:

TTS Base Model
↓
TTS v1
↓
Human Listening
↓
New Correction Recordings
↓
Retained Correction Dataset
↓
TTS v2 Fine-Tuning
↓
Updated TTS Model

The corrected training examples were manually checked after v2 training and the target Tamil pronunciations were observed to be correct for those examples.

## 11. TTS Evaluation

Held-out TTS test samples:

1. வணக்கம் அருண்
2. ₹25,500
3. Veltech AI 2026

Final evaluation command:

python scripts/evaluate_tts.py \
  --model models/tts/tam/v2 \
  --manifest data/tts/test.jsonl \
  --output-dir reports/tts_final \
  --report reports/tts_final.json

The evaluation generated three audio outputs.

No automatic numeric TTS quality metric is used in this experiment. Pronunciation correctness, intelligibility and artifacts require human listening.

## 12. Training Data Persistence

STT:
data/stt/corrections.jsonl

TTS:
data/tts/corrections.jsonl

New corrections are added to the existing dataset. Historical correction records are retained.

## 13. Model Versioning

models/
├── stt/
│   └── v1/
└── tts/
    └── tam/
        ├── v1/
        └── v2/

Each trained version is stored separately.

## 14. Model Reusability

Saved artifacts include:
- model.safetensors
- config.json
- tokenizer configuration
- processor/tokenizer files where applicable
- training_manifest.json

The saved models can be loaded again for inference.

## 15. Trainable and Frozen Parameters

The implemented training scripts optimize model parameters from the pretrained checkpoints.

No specific model layers are intentionally frozen. Therefore, the project does not claim that particular encoder, decoder or VITS layers are frozen.

## 16. Test Data Separation

STT training:
RECORD_AUDIO_001.wav
RECORD_AUDIO_002.wav
RECORD_AUDIO_003.wav

Held-out STT testing:
TEST_TAMIL_001.wav
TEST_NUMBERS_001.wav
TEST_ALPHANUMERIC_001.wav

Held-out test recordings are not used as training corrections.

## 17. Input Coverage

The project includes:
- Tamil words and sentences
- Names
- English text
- Numbers
- Dates
- Alphanumeric values
- Symbols
- Special characters
- Unicode text

Examples:
- வணக்கம் கவின்
- நான் இன்று அலுவலகத்திற்கு செல்கிறேன்.
- Order 123 is ready on 26/09/2026.
- AI-2026 Veltech.AI
- ₹25,500

## 18. Model-Level Learning

The project does not use:
- Hardcoded string replacements
- Dictionary replacement rules
- If/else correction rules
- Post-processing replacement tables

Instead:

Incorrect Model Output
↓
Human Correction
↓
Persistent Training Record
↓
Model Fine-Tuning
↓
New Model Version
↓
Retest

## 19. Reproducibility

- Python: 3.14.4
- PyTorch: 2.14.0+cpu
- Transformers: 5.17.0
- STT: openai/whisper-small
- TTS: facebook/mms-tts-tam
- STT learning rate: 1e-5
- TTS learning rate: 1e-6
- STT model: v1
- TTS models: v1 and v2
- Hardware: CPU-only
- STT metric: WER
- TTS evaluation: generated held-out audio + human listening

## 20. Final Results

STT:
- Base model: openai/whisper-small
- Fine-tuned model: models/stt/v1
- Held-out samples: 3
- Final WER: 1.2000
- No measurable held-out improvement claimed.

TTS:
- Base model: facebook/mms-tts-tam
- Initial model: models/tts/tam/v1
- Continued-learning model: models/tts/tam/v2
- Two new pronunciation corrections were incorporated.
- Corrected training examples were manually checked.
- Final held-out evaluation generated three audio samples for qualitative review.

## 21. Limitations

- Very small speech dataset.
- STT held-out WER did not improve.
- TTS quality requires human listening.
- Larger and more diverse data is required for production-quality performance.
- More compute would help stronger multilingual training.
- Current TTS training uses an exercise-oriented spectrogram reconstruction approach because direct VITS training is not supported by the installed Transformers interface.

## 22. Conclusion

The project demonstrates an end-to-end correction-driven speech model training workflow.

Human corrections become persistent training data and are incorporated through model-level fine-tuning instead of hardcoded application-level corrections.

The project produces reusable versioned STT and TTS model artifacts and demonstrates continued learning through the TTS v1 → v2 workflow.