** Run instruction **
1. create venv
   python -m venv env
2. Activate venv
   .\env\Scripts\activate
3. Insall packages
pip install -r requirements.txt
4. Paste your openai key in .env file
5. Run main.py file

.env variables to be used:
1. INPUT_DEVICE_INDEX - index of the microphone you want to use. By default 1
2. AUDIO_MODEL - model you want to use.
   - openai - API whisper from openai - by default
   - scalepoint - use API from scalepoint
   - scalepoint_translation - use API from scalepoint with translation mode
   - openai/whisper-tiny.en - openai whisper from hugging_face
   - faster-whisper - small model from faster-whisper library
   - google-cloud-speech - speech recognition model from google
   - elevenlabs - speech recognition model from elevenlabs

python main.py
runs entire app translator_app - models to transcribe and translate
 (lub )

**files**

TranslatorApp - tkinter app frontend
__init__.py - parameters setup - rate, chunk
Utils - logic of models and queues running

**Test recordings**
https://drive.google.com/drive/folders/1nbRGH0geoiGHRPs5sZK2aKKcOhXsw9nb

** Ideas to add **

https://huggingface.co/nvidia/parakeet-rnnt-1.1b - local nvidia model

polish model: asr_model_pl = nemo_asr.models.**EncDecCTCModel**.from_pretrained(model_name="**stt_pl_quartznet15x5**")


Known issues:
   Transcription:
1. Silence problem - in the silence there are hallucinations - ex. Thank you for your attention
2. Audio chunking optimization - too long - too big delay. Too short - inaccurate results - not enough context. We agreed to 5 second sweet-spot.
4. Translation - Sometimes inaccurate - context needs to be added.