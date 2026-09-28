import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa
import torch
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    required=True
)

parser.add_argument(
    "--base-model",
    default="openai/whisper-small"
)

parser.add_argument(
    "--output",
    required=True
)

parser.add_argument(
    "--epochs",
    type=int,
    default=3
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-5
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=2
)

parser.add_argument(
    "--gradient-accumulation-steps",
    type=int,
    default=4
)

args = parser.parse_args()


# --------------------------------------------------
# Load training records
# --------------------------------------------------

records = load_records(args.dataset)

missing = [
    r["audio"]
    for r in records
    if not os.path.exists(r["audio"])
]

if missing:
    raise FileNotFoundError(
        "Missing audio files: " + ", ".join(missing)
    )


print(f"Training samples: {len(records)}")


# --------------------------------------------------
# Load Whisper processor
# --------------------------------------------------

processor = WhisperProcessor.from_pretrained(
    args.base_model
)


# --------------------------------------------------
# Load audio manually
# This avoids datasets.Audio / torchcodec
# --------------------------------------------------

prepared_records = []

for record in records:

    audio, sampling_rate = librosa.load(
        record["audio"],
        sr=16000,
        mono=True
    )

    input_features = processor.feature_extractor(
        audio,
        sampling_rate=16000
    ).input_features[0]

    labels = processor.tokenizer(
        record["corrected_text"]
    ).input_ids

    prepared_records.append(
        {
            "input_features": input_features,
            "labels": labels
        }
    )


# --------------------------------------------------
# Create Hugging Face Dataset
# --------------------------------------------------

dataset = Dataset.from_list(prepared_records)


# --------------------------------------------------
# Data collator
# --------------------------------------------------

@dataclass
class Collator:

    processor: Any

    def __call__(
        self,
        features: List[
            Dict[
                str,
                Union[
                    List[int],
                    torch.Tensor
                ]
            ]
        ]
    ):

        input_features = [
            {
                "input_features": f["input_features"]
            }
            for f in features
        ]

        label_features = [
            {
                "input_ids": f["labels"]
            }
            for f in features
        ]

        x = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        y = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        x["labels"] = y["input_ids"].masked_fill(
            y["attention_mask"].ne(1),
            -100
        )

        return x


# --------------------------------------------------
# Load Whisper model
# --------------------------------------------------

model = WhisperForConditionalGeneration.from_pretrained(
    args.base_model
)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

training = Seq2SeqTrainingArguments(

    output_dir=args.output,

    per_device_train_batch_size=args.batch_size,

    gradient_accumulation_steps=args.gradient_accumulation_steps,

    learning_rate=args.learning_rate,

    num_train_epochs=args.epochs,

    warmup_steps=50,

    fp16=torch.cuda.is_available(),

    logging_steps=10,

    save_strategy="epoch",

    report_to="none",

    remove_unused_columns=False
)


# --------------------------------------------------
# Trainer
# --------------------------------------------------

trainer = Seq2SeqTrainer(

    model=model,

    args=training,

    train_dataset=dataset,

    data_collator=Collator(processor)
)


# --------------------------------------------------
# Fine-tuning
# --------------------------------------------------

print()
print("=" * 60)
print("Starting Whisper fine-tuning")
print("=" * 60)

trainer.train()


# --------------------------------------------------
# Save trained model
# --------------------------------------------------

trainer.save_model(args.output)

processor.save_pretrained(args.output)


# --------------------------------------------------
# Save training manifest
# --------------------------------------------------

manifest = {

    "base_model": args.base_model,

    "dataset": args.dataset,

    "epochs": args.epochs,

    "learning_rate": args.learning_rate,

    "batch_size": args.batch_size,

    "gradient_accumulation_steps":
        args.gradient_accumulation_steps,

    "records": len(records)

}


with open(
    os.path.join(
        args.output,
        "training_manifest.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 60)
print("Training completed")
print("Saved:", args.output)
print("=" * 60)