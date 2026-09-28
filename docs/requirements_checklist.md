# STT & TTS Model Training – Requirements Checklist

## 1. Model Selection
- [x] Free/open-source STT model selected: openai/whisper-small
- [x] Free/open-source TTS model selected: facebook/mms-tts-tam
- [x] Models are used for inference and model-level fine-tuning.

## 2. Input and Output Coverage
- [x] Words
- [x] Sentences
- [x] Names
- [x] Numbers
- [x] Dates
- [x] Alphanumeric values
- [x] Symbols / special characters
- [x] Unicode text
- [x] Indian-language examples, including Tamil
- [x] English examples

## 3. Multilingual Support
- [x] Multilingual STT base model selected.
- [x] Tamil STT examples tested.
- [x] English STT examples tested.
- [x] Tamil TTS model selected and tested.
- [ ] All requested Indian languages were not fully trained/evaluated in this small CPU experiment.

## 4. Error Correction and Learning
- [x] STT incorrect outputs are stored with human corrections.
- [x] TTS pronunciation corrections are stored with human reference recordings.
- [x] Corrections are stored in persistent JSONL datasets.
- [x] Corrections are used for model-level fine-tuning.
- [x] No hardcoded replacement dictionary is used for learning.

## 5. Model-Level Training
- [x] STT Whisper model fine-tuned using correction data.
- [x] TTS MMS/VITS model fine-tuned using text + reference audio.
- [x] Training updates model parameters.
- [x] TTS uses model-level spectrogram reconstruction fine-tuning because the installed Transformers VITS training interface does not support direct VITS training.

## 6. Training Data Persistence
- [x] STT correction dataset retained.
- [x] TTS correction dataset retained.
- [x] Historical correction records are not intentionally deleted.
- [x] New corrections are added to the existing dataset.
- [x] Held-out test data is kept separate from training corrections.

## 7. Continued Learning
- [x] Initial TTS model v1 trained.
- [x] Human pronunciation corrections collected after v1.
- [x] New correction recordings added to the retained dataset.
- [x] TTS v2 trained from TTS v1 using the complete retained correction dataset.
- [x] Corrected training examples were manually checked after v2.
- [x] Continued-learning workflow demonstrated.
- [ ] Quantitative TTS improvement metric is not available in this experiment.

## 8. Model Reusability
- [x] STT v1 saved under models/stt/v1.
- [x] TTS v1 saved under models/tts/tam/v1.
- [x] TTS v2 saved under models/tts/tam/v2.
- [x] Model configuration files saved.
- [x] Tokenizer/processor supporting files saved.
- [x] model.safetensors saved.
- [x] training_manifest.json saved.

## 9. Evaluation
- [x] STT held-out test set created.
- [x] STT baseline evaluated.
- [x] STT fine-tuned model evaluated.
- [x] Final STT WER recorded: 1.2000 on 3 held-out samples.
- [x] TTS held-out evaluation generated 3 audio samples.
- [x] TTS corrected training examples were manually listened to.
- [ ] Final held-out TTS samples still require/benefit from final human listening review.

## 10. Expected Demonstration
- [x] Base Model
- [x] Incorrect Output
- [x] Human Correction
- [x] Training Data
- [x] Fine-Tuning
- [x] Updated Model
- [x] Retest
- [x] Continued-learning demonstration
- [x] Before/after workflow documented
- [x] Actual measured STT result documented without fabricated improvement.

## 11. Expected Deliverables
- [x] STT inference and training code
- [x] TTS inference and training code
- [x] Correction datasets
- [x] Saved STT model
- [x] Saved TTS models
- [x] Supporting tokenizer/configuration files
- [x] Evaluation reports
- [x] Technical documentation
- [x] Continued-learning demonstration