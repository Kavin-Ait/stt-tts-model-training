"""MMS/VITS spectrogram-loss fine-tuning.

Uses the pretrained MMS/VITS generator and updates its parameters using
text + real reference speech. Transformers' built-in VITS training path
is not used because it currently raises NotImplementedError.
"""

import argparse
import json
import os
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
from transformers import VitsModel, VitsTokenizer


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)

parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--learning-rate", type=float, default=1e-6)
parser.add_argument("--sample-rate", type=int, default=16000)

args = parser.parse_args()


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

records = [
    json.loads(line)
    for line in open(args.dataset, encoding="utf-8")
    if line.strip()
]

for record in records:
    if not os.path.exists(record["reference_audio"]):
        raise FileNotFoundError(record["reference_audio"])


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Training samples:", len(records))


# ---------------------------------------------------------
# Load pretrained MMS/VITS model
# ---------------------------------------------------------

tokenizer = VitsTokenizer.from_pretrained(args.model)

model = VitsModel.from_pretrained(args.model).to(device)

model.train()


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
)


# ---------------------------------------------------------
# Spectrogram settings
# ---------------------------------------------------------

n_fft = int(getattr(model.config, "fft_size", 1024))
hop_length = int(getattr(model.config, "hop_length", 256))
win_length = int(getattr(model.config, "win_length", 1024))

window = torch.hann_window(
    win_length,
    device=device,
)


# ---------------------------------------------------------
# Reference audio → spectrogram
# ---------------------------------------------------------

def load_reference_spectrogram(path):

    waveform, sample_rate = librosa.load(
        path,
        sr=args.sample_rate,
        mono=True,
    )

    waveform = torch.tensor(
        waveform,
        dtype=torch.float32,
        device=device,
    )

    spectrogram = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    ).abs()

    return spectrogram


# ---------------------------------------------------------
# Generated waveform → spectrogram
# ---------------------------------------------------------

def waveform_to_spectrogram(waveform):

    spectrogram = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    ).abs()

    return spectrogram


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

loss_history = []


for epoch in range(args.epochs):

    model.train()

    total_loss = 0.0

    print()
    print(f"Epoch {epoch + 1}/{args.epochs}")

    for index, record in enumerate(records, start=1):

        optimizer.zero_grad()

        # Text → tokens
        inputs = tokenizer(
            record["text"],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Model → generated waveform
        output = model(**inputs)

        generated_waveform = output.waveform

        # Reference audio → target spectrogram
        target_spectrogram = load_reference_spectrogram(
            record["reference_audio"]
        )

        # Generated waveform → predicted spectrogram
        predicted_spectrogram = waveform_to_spectrogram(
            generated_waveform
        )

        # Align target spectrogram with generated spectrogram
        target_spectrogram = F.interpolate(
            target_spectrogram.unsqueeze(0).unsqueeze(0),
            size=(
                predicted_spectrogram.shape[0],
                predicted_spectrogram.shape[1],
            ),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Model-level reconstruction loss
        loss = F.l1_loss(
            predicted_spectrogram,
            target_spectrogram,
        )

        # Backpropagation
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        loss_value = float(
            loss.detach().cpu()
        )

        total_loss += loss_value

        print(
            f"  Sample {index}/{len(records)} "
            f"loss={loss_value:.6f}"
        )

    epoch_loss = total_loss / max(1, len(records))

    loss_history.append(
        {
            "epoch": epoch + 1,
            "loss": epoch_loss,
        }
    )

    print(
        f"Epoch {epoch + 1} average loss: "
        f"{epoch_loss:.6f}"
    )


# ---------------------------------------------------------
# Save trained model
# ---------------------------------------------------------

output_path = Path(args.output)

output_path.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(output_path)

tokenizer.save_pretrained(output_path)


# ---------------------------------------------------------
# Save training metadata
# ---------------------------------------------------------

training_manifest = {
    "base_model": args.model,
    "dataset": args.dataset,
    "epochs": args.epochs,
    "learning_rate": args.learning_rate,
    "sample_rate": args.sample_rate,
    "records": len(records),
    "loss_history": loss_history,
    "training_method": "model-level spectrogram reconstruction fine-tuning",
    "note": (
        "Pretrained MMS/VITS parameters were updated using "
        "text-conditioned generated waveform spectrograms "
        "against real reference speech."
    ),
}


with open(
    output_path / "training_manifest.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        training_manifest,
        file,
        ensure_ascii=False,
        indent=2,
    )


print()
print("Saved TTS model:", output_path)