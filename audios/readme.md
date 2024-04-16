



**Aby uruchomić **
1. Stworz zmienna srodowiskowa 
   python -m venv env
2. Uaktywnij 
   .\env\Scripts\activate
3. Instalacja pakietow 
pip install -r requirements.txt
4. Podmien klucz openai w env 
5. Uruchom skrypt main.py

.env variables to be used:
1. INPUT_DEVICE_INDEX - index of the microphone you want to use. By default 1
2. AUDIO_MODEL - model you want to use. 
   - openai - API whisper from openai - by default
   - scalepoint - use API from scalepoint
   - scalepoint_translation - use API from scalepoint with translation mode
   - openai/whisper-tiny.en - openai whisper from hugging_face
   - faster-whisper - small model from faster-whisper library

python main.py
bazuje o pliki z folderu  translator_app -  whisper do transkrypcji + translacji 
 (lub ) 
 
 python main-1.py 
 ( w skrypcie widze, ze Bartek probowal dodac raga )


**Pliki** 

TranslatorApp - wyglad aplikacji 
__init__.py - najwazniejsze parametry - rate, chunk
Utils - model whisper 

 

 

**Nagrywki / Testy** 
https://drive.google.com/drive/folders/1nbRGH0geoiGHRPs5sZK2aKKcOhXsw9nb 

** Do rozwazania zamiast whispera **

https://huggingface.co/nvidia/parakeet-rnnt-1.1b

MODEL POLSKI: asr_model_pl = nemo_asr.models.**EncDecCTCModel**.from_pretrained(model_name="**stt_pl_quartznet15x5**")


Znane problemy:
   Tranksrypcja:
1. Problem ciszy - w trakice ciszy pojawiaja sie dziwne zwroty - np. Allah akbar    
2. Problem przeladowanych promptow - zbyt dlugie prompty zwracaja pozniej informacje, chcemy aby komunikaty wyswietlaly sie w max 5sekund
3. Gubiony kontekst jesli zbyt krotkie chunki sa wysylane do modelu (2-3s)
   1. Mozna sparalelizowac wysylke chunkow. Przykladowo dla pliku audio 10sekundowego, bylo by to 5 chunkow, kazdy wysylany co 1.5sekundy 
   
4. Translacja - Obecnei translacja realizowana przez whisper. Lepiej sobie radzi niz GPT 4. Do porownania, czy lepiej niz 3.5 Kwestia nadania kontekstu tlumaczeniom. "widze dwa komunikaty" - I see two communists.