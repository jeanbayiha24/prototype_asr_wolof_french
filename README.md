# prototype_asr_wolof_french : Wolof ASR + Translation Demo

A simple **Streamlit web application** that allows you to:

- Record or upload audio in **Wolof** language  
- Transcribe it using two different ASR models  
- Automatically translate the transcription from **Wolof → French**  
- Keep history of the last transcriptions & translations

## Available Models

### Speech Recognition (ASR)

| Model on HuggingFace                          | Description                                      | Type       | Speed / Accuracy (approx.)         |
|-----------------------------------------------|--------------------------------------------------|------------|------------------------------------|
| `speechbrain/asr-wav2vec2-dvoice-wolof`       | Wav2Vec2 + CTC fine-tuned on DVoice corpus       | CTC        | Faster ∙ Good on clean speech      |
| `dofbi/wolof-asr`                             | Whisper-small strongly fine-tuned on Wolof       | Transformer| Slower ∙ Currently best accuracy   |

### Translation (Wolof ↔ French)

| Model on HuggingFace                          | Description                                                                 | Base Model                          | BLEU (approx.)* | Direction                  |
|-----------------------------------------------|-----------------------------------------------------------------------------|-------------------------------------|-----------------|----------------------------|
| `galsenai/wolofToFrenchTranslator_nllb`       | Fine-tuned NLLB for high-quality Wolof–French translation                  | facebook/nllb-200-distilled-600M    | ~13             | Wolof → French (main) + French → Wolof |


> The translation model was fine-tuned by **GalsenAI** using manually aligned parallel corpora, Common Voice, Wikipedia, administrative documents, and data collected via LinguaSprint Africa.

## Features

- Microphone recording directly in browser  
- Audio file upload (.wav, .mp3, .ogg, .m4a)  
- Automatic audio resampling (→16kHz) & mono conversion when needed  
- GPU acceleration (`cuda`) when available  
- Simple translation interface (currently Wolof → French by default)  
- Last 10 transcriptions + translations history  
- Temporary audio files are automatically deleted

## Quick Start

```bash
# Recommended: virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# or venv\Scripts\activate        # Windows

or use a conda virtual environment

pip install -r requirements.txt

streamlit run app.py
```
First run will download ~2.5–3.5 GB of models (SpeechBrain + Whisper-small + NLLB-600M fine-tuned).

## Current Limitations

- No real-time/streaming transcription (processes complete utterances)
- Translation quality depends heavily on transcription accuracy
- Noisy audio → degraded performance 
- French → Wolof translation is technically supported by the model but not yet exposed in the UI

## Acknowledgments
Huge thanks to the creators of these very nice open models:
- [speechbrain/asr-wav2vec2-dvoice-wolof](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-wolof)
- [dofbi/wolof-asr](https://huggingface.co/dofbi/wolof-asr)
- [galsenai/wolofToFrenchTranslator_nllb](https://huggingface.co/galsenai/wolofToFrenchTranslator_nllb) — GalsenAI Lab
