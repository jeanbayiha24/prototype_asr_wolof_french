import streamlit as st
import os
import time
from datetime import datetime
#import sqlite3
from speechbrain.inference.ASR import EncoderASR #type: ignore
import torch
import torchaudio
import torchaudio.transforms as T
# Import Whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
#from langchain_ollama import ChatOllama
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#On save le model en local a la 1ere execution
SAVEDIR = "pretrained_models/asr-wav2vec2-dvoice-wolof"
# Modèles dispo
MODELS = {
    "speechbrain/asr-wav2vec2-dvoice-wolof": "SpeechBrain Wav2Vec2-DVoice",
    "dofbi/wolof-asr": "Whisper-small fine-tuné Wolof"
}


#load the wolof asr speechbrain model
@st.cache_resource #mettre en cache le modele pour ne plus le recharger a chaque fois
def load_speechbrain():
    return EncoderASR.from_hparams(
        source= "speechbrain/asr-wav2vec2-dvoice-wolof",
        savedir=SAVEDIR,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

#load the fine-tuned whisper
@st.cache_resource
def load_whisper():
    model_name = "dofbi/wolof-asr"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

st.title("Test ASR Wolof (SpeechBrain & Whisper) & Traduction Wolof-Français")

#selection modele
selected_model = st.radio("Choisis le modèle ASR :", 
                          options=list(MODELS.keys()), 
                          index=0, 
                          format_func=lambda x: MODELS[x]
                          )

# Chargement du modèle sélectionné
if selected_model == "speechbrain/asr-wav2vec2-dvoice-wolof":
    asr_model = load_speechbrain()
    model_type = "speechbrain"
    st.info("Modèle chargé : SpeechBrain Wav2Vec2-DVoice (CTC)")
else:
    whisper_processor, whisper_model = load_whisper()
    model_type = "whisper"
    st.info("Modèle chargé : Whisper-small fine-tuné Wolof")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#checkpoint du nllb pour la traduction
checkpoint= "galsenai/wolofToFrenchTranslator_nllb"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#galsenai translation model
translator_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

@st.cache_resource
def translate(text, lang):
    if lang.lower() == "wo":
        prefix = "translate Wolof to French: "
    elif lang.lower() == "fr":
        prefix = "translate French to Wolof: "
    else:
        raise ValueError("Invalid language code")
    inputs = tokenizer(prefix + text, return_tensors="pt").to(device)
    translated_tokens = translator_model.generate(**inputs, max_length=30)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]




#Onglets pour les methodes de recuperation de l'audio
tab1, tab2 = st.tabs(["Enregistrement micro", "Upload fichier audio"])

transcription = None
audio_path_play = None #pour pouvoir ecouter l'audio

def transcribe_file(temp_path, model_type):
    try:
        if model_type == "speechbrain": 
            transcription = asr_model.transcribe_file(temp_path).lower()
        else:#whisper
            waveform, sample_rate = torchaudio.load(temp_path)
            #resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            #whisper attend des audios en mono
            waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze()

            inputs = whisper_processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
            predicted_ids = whisper_model.generate(inputs.to(whisper_model.device))
            transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower()
        st.success(f"Transcription: {transcription}")
        return transcription
            
    except Exception as e:
        st.error(f"Erreur lors de la transcription : {e}")
        return None
    


#tab1 : micro
with tab1:
    st.write("Parle en wolof pres du micro :")
    audio = st.audio_input("Clique pour enregistrer")
    if audio is not None:
        temp_path = "temp_micro.wav"
        with open(temp_path, "wb") as f:
            f.write(audio.getvalue())
        
        #afficher l'audio pour permettre de l'ecouter
        st.audio(temp_path, format="audio/wav")

        #transcription
        transcription = transcribe_file(temp_path, model_type)
        #On garde le path pour l'ecoute
        audio_path_play = temp_path
    else:
        audio_path_play = None

#tab2 : upload fichier audio
with tab2:
    uploaded_file = st.file_uploader("Choisis un fichier audio en wolof (.wav, .mp3 ou .ogg)", type=["wav", "mp3", "ogg", "m4a"])
    if uploaded_file is not None:
        temp_path = f"temp_upload_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        #afficher l'audio pour permettre de l'ecouter
        st.audio(temp_path, format="audio/wav")


        transcription = transcribe_file(temp_path, model_type)
        #On garde le path pour l'ecoute
        audio_path_play = temp_path
    else:
        audio_path_play = None

#On traduit la transcription si elle existe
traduction = None
if transcription:
    traduction = translate(transcription, lang="wo")
    st.success(f"Traduction français : {traduction}")


#On garde l'historique des transcriptions
if 'history' not in st.session_state:
    st.session_state.history = []

if transcription and traduction:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_label = MODELS[selected_model]
    st.session_state.history.append(f"{timestamp} | {model_label} : {transcription} -> {traduction}")
    
    #supprimer le fichier temporaire pour ne pas surcharger le stockage
    if audio_path_play and os.path.exists(audio_path_play):
        try:
            os.remove(audio_path_play)
        except Exception as e:
            pass


st.subheader("Historique des transcriptions")#les 10 dernieres transcriptions
for item in st.session_state.history[-10:]:
    st.write(item)
