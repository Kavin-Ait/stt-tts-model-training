#!/bin/bash
set -e

echo "======================================"
echo " STT/TTS PROJECT FINALIZATION"
echo "======================================"

echo
echo "[1/6] Removing temporary TTS test audio..."
rm -f data/tts/audio/TTS_TRAIN_TEST_001.wav
rm -f data/tts/audio/TTS_EASY_PRONUNCIATION_TEST.wav
rm -f data/tts/audio/TTS_FULL_SENTENCE_TEST.wav
rm -f data/tts/audio/TTS_OUTPUT_004.wav
rm -f data/tts/audio/TTS_OUTPUT_004_v2.wav

echo
echo "[2/6] Checking final TTS audio..."
test -f data/tts/audio/TTS_FINAL_TEST.wav
file data/tts/audio/TTS_FINAL_TEST.wav

echo
echo "[3/6] Checking STT model..."
test -f models/stt/v1/model.safetensors
test -f models/stt/v1/config.json
test -f models/stt/v1/training_manifest.json

echo
echo "[4/6] Checking TTS v1 and v2 models..."
test -f models/tts/tam/v1/model.safetensors
test -f models/tts/tam/v2/model.safetensors

echo
echo "[5/6] Validating JSONL files..."
for file in \
    data/stt/train.jsonl \
    data/stt/corrections.jsonl \
    data/tts/test.jsonl \
    data/tts/corrections.jsonl
do
    echo "Checking: $file"

    while IFS= read -r line; do
        echo "$line" | python -m json.tool > /dev/null
    done < "$file"

    echo "OK: $file"
done

echo
echo "[6/6] Final project status..."
echo
git status --short

echo
echo "======================================"
echo " FINAL MODEL SUMMARY"
echo "======================================"
echo "STT : models/stt/v1"
echo "TTS : models/tts/tam/v1"
echo "TTS : models/tts/tam/v2"
echo "FINAL TTS AUDIO : data/tts/audio/TTS_FINAL_TEST.wav"

echo
echo "======================================"
echo " FINALIZATION CHECK COMPLETED"
echo "======================================"
echo
echo "Review the git status above."
echo "If everything looks correct, run:"
echo
echo "git add data/stt data/tts models/README.md"
echo 'git commit -m "Finalize STT and TTS training updates"'
echo "git push origin main"
