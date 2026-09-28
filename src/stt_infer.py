import argparse

import librosa
import torch

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--audio",
    required=True,
    help="Path to input WAV audio"
)

parser.add_argument(
    "--language",
    default="en",
    help="Whisper language code, example: ta, en, hi"
)

parser.add_argument(
    "--model",
    default="openai/whisper-small",
    help="Whisper model name or local trained model path"
)

args = parser.parse_args()


# --------------------------------------------------
# Load model and processor
# --------------------------------------------------

print()
print("=" * 60)
print("Loading Whisper model")
print("=" * 60)

print("Model:", args.model)
print("Audio:", args.audio)
print("Language:", args.language)

processor = WhisperProcessor.from_pretrained(
    args.model
)

model = WhisperForConditionalGeneration.from_pretrained(
    args.model
)


# --------------------------------------------------
# Select device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

print("Device:", device)


# --------------------------------------------------
# Load audio
# --------------------------------------------------

audio, sampling_rate = librosa.load(
    args.audio,
    sr=16000,
    mono=True
)


# --------------------------------------------------
# Prepare audio features
# --------------------------------------------------

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

input_features = inputs["input_features"].to(device)


# --------------------------------------------------
# Language / transcription prompt
# --------------------------------------------------

forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=args.language,
    task="transcribe"
)


# --------------------------------------------------
# Generate transcription
# --------------------------------------------------

with torch.no_grad():

    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids
    )


# --------------------------------------------------
# Decode result
# --------------------------------------------------

text = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("=" * 60)
print("Transcription Result")
print("=" * 60)

print("Transcription:", text)

print("=" * 60)