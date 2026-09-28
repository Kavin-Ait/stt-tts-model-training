"""TTS evaluation: generate held-out test utterances and create a manifest.
Human listening/quality assessment is required because waveform similarity alone is not a reliable TTS quality metric."""
import argparse,json,os
from pathlib import Path
import torch,soundfile as sf
from transformers import VitsModel,VitsTokenizer,set_seed
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--manifest',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--report',required=True); args=p.parse_args()
records=[json.loads(x) for x in open(args.manifest,encoding='utf-8') if x.strip()]; Path(args.output_dir).mkdir(parents=True,exist_ok=True)
tok=VitsTokenizer.from_pretrained(args.model); model=VitsModel.from_pretrained(args.model); model.eval(); set_seed(555); results=[]
for i,r in enumerate(records,1):
    x=tok(r['text'],return_tensors='pt')
    with torch.no_grad(): wav=model(**x).waveform[0].cpu().numpy()
    out=str(Path(args.output_dir)/f'{i:03d}.wav'); sf.write(out,wav,model.config.sampling_rate); results.append({'text':r['text'],'language':r['language'],'audio':out})
Path(args.report).parent.mkdir(parents=True,exist_ok=True); json.dump({'model':args.model,'samples':results,'human_review_required':True},open(args.report,'w',encoding='utf-8'),ensure_ascii=False,indent=2); print('Generated',len(results),'TTS samples.')
