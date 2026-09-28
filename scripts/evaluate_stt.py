import argparse
import json
import os
from pathlib import Path

import jiwer
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


records = [
    json.loads(x)
    for x in open(args.manifest, encoding="utf-8")
    if x.strip()
]

missing = [
    r["audio"]
    for r in records
    if not os.path.exists(r["audio"])
]

if missing:
    raise FileNotFoundError(", ".join(missing))


processor = WhisperProcessor.from_pretrained(args.model)
model = WhisperForConditionalGeneration.from_pretrained(args.model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


predictions = []
references = []


for record in records:

    audio, sampling_rate = librosa.load(
        record["audio"],
        sr=16000,
        mono=True
    )

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=record.get("language", "en"),
        task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"].to(device),
            forced_decoder_ids=forced_decoder_ids
        )

    prediction = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    predictions.append(prediction)
    references.append(record["reference"])

    print(f"Audio      : {record['audio']}")
    print(f"Reference  : {record['reference']}")
    print(f"Prediction : {prediction}")
    print("-" * 60)


wer = jiwer.wer(references, predictions)


output = {
    "model": args.model,
    "samples": len(records),
    "wer": wer,
    "predictions": [
        {
            "reference": reference,
            "prediction": prediction
        }
        for reference, prediction in zip(references, predictions)
    ]
}


Path(args.output).parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    args.output,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        output,
        file,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 60)
print(f"Baseline WER : {wer:.4f}")
print(f"Samples      : {len(records)}")
print(f"Output       : {args.output}")
print("=" * 60)