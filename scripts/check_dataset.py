import json,os,sys
errors=[]
for path,fields in [('data/stt/corrections.jsonl',['audio','language','corrected_text']),('data/stt/test.jsonl',['audio','language','reference']),('data/tts/corrections.jsonl',['text','language','reference_audio']),('data/tts/test.jsonl',['text','language'])]:
    if not os.path.exists(path): errors.append(f'missing {path}'); continue
    for n,line in enumerate(open(path,encoding='utf-8'),1):
        if not line.strip(): continue
        r=json.loads(line)
        for f in fields:
            if not r.get(f): errors.append(f'{path}:{n}: missing {f}')
        for f in ('audio','reference_audio'):
            if f in r and r[f] and not os.path.exists(r[f]) and 'RECORD_' not in r[f] and 'TEST_' not in r[f]: errors.append(f'{path}:{n}: missing file {r[f]}')
if errors:
    print('\n'.join(errors)); sys.exit(1)
print('Dataset manifests are structurally valid. Add/record the referenced audio before training.')
