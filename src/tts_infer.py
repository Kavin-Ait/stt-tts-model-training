import argparse, torch, soundfile as sf
from transformers import VitsModel, VitsTokenizer, set_seed

LANG_MODELS={
 'tam':'facebook/mms-tts-tam','hin':'facebook/mms-tts-hin','tel':'facebook/mms-tts-tel','kan':'facebook/mms-tts-kan',
 'mal':'facebook/mms-tts-mal','ben':'facebook/mms-tts-ben','mar':'facebook/mms-tts-mar','guj':'facebook/mms-tts-guj','pan':'facebook/mms-tts-pan'
}
p=argparse.ArgumentParser(); p.add_argument('--text',required=True); p.add_argument('--language',required=True); p.add_argument('--model'); p.add_argument('--output',default='output.wav'); args=p.parse_args()
model_id=args.model or LANG_MODELS[args.language]
tokenizer=VitsTokenizer.from_pretrained(model_id); model=VitsModel.from_pretrained(model_id)
inputs=tokenizer(args.text, return_tensors='pt')
set_seed(555)
with torch.no_grad(): out=model(**inputs).waveform[0].cpu().numpy()
sf.write(args.output,out,model.config.sampling_rate)
print(f'Generated {args.output} using {model_id}')
