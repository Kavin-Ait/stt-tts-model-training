# TTS dataset

Each correction requires a real reference recording of the desired pronunciation.
Use consistent speaker, microphone, sample rate and recording conditions for a controlled experiment.

Each JSONL record must contain:
- `text`
- `language`: MMS ISO-639-3 code
- `reference_audio`

The TTS test manifest is kept separate from training data.
