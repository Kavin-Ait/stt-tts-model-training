import json,tempfile
from pathlib import Path
import streamlit as st
st.set_page_config(page_title='Veltech.AI STT & TTS Training',layout='wide')
st.title('Veltech.AI – STT & TTS Correction Training Demo')
st.caption('Human corrections are persisted as training data. No inference-time hardcoded replacement is used.')

def append(path,record):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a',encoding='utf-8') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')

t1,t2=st.tabs(['STT correction','TTS correction'])
with t1:
    st.subheader('Speech → Text → Human correction')
    audio=st.file_uploader('Upload speech recording',type=['wav','mp3','m4a'],key='stt_audio')
    lang=st.text_input('Whisper language code','ta',key='stt_lang')
    output=st.text_area('Current model output',key='stt_output')
    corrected=st.text_area('Human corrected transcript',key='stt_corrected')
    if st.button('Save STT correction'):
        if not audio or not corrected: st.error('Audio and corrected transcript are required.')
        else:
            path=Path('data/stt/audio')/audio.name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(audio.getbuffer())
            append('data/stt/corrections.jsonl',{'audio':str(path),'language':lang,'model_output':output,'corrected_text':corrected,'source':'streamlit_human_correction'})
            st.success('Saved as persistent training data.')
with t2:
    st.subheader('Text → Speech → Human reference pronunciation')
    text=st.text_area('Text',key='tts_text'); lang=st.text_input('MMS language code','tam',key='tts_lang'); ref=st.file_uploader('Upload corrected reference pronunciation',type=['wav'],key='tts_ref')
    if st.button('Save TTS correction'):
        if not text or not ref: st.error('Text and reference audio are required.')
        else:
            path=Path('data/tts/audio')/ref.name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(ref.getbuffer())
            append('data/tts/corrections.jsonl',{'text':text,'language':lang,'reference_audio':str(path),'source':'streamlit_human_reference'})
            st.success('Saved as persistent training data.')
