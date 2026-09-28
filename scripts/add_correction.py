import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='kind',required=True)
a=sub.add_parser('stt'); a.add_argument('--audio',required=True); a.add_argument('--language',required=True); a.add_argument('--model-output',default=''); a.add_argument('--corrected-text',required=True)
b=sub.add_parser('tts'); b.add_argument('--text',required=True); b.add_argument('--language',required=True); b.add_argument('--reference-audio',required=True)
args=p.parse_args(); Path('data/'+args.kind).mkdir(parents=True,exist_ok=True)
if args.kind=='stt':
    if not os.path.exists(args.audio): raise FileNotFoundError(args.audio)
    r={'audio':args.audio,'language':args.language,'model_output':args.model_output,'corrected_text':args.corrected_text,'source':'human_correction'}; out='data/stt/corrections.jsonl'
else:
    if not os.path.exists(args.reference_audio): raise FileNotFoundError(args.reference_audio)
    r={'text':args.text,'language':args.language,'reference_audio':args.reference_audio,'source':'human_reference'}; out='data/tts/corrections.jsonl'
with open(out,'a',encoding='utf-8') as f: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('Appended:',out)
