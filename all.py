import pyaudio
import wave
import librosa
import os
import numpy as np
import threading
import logging
import queue
import openai
from dotenv import load_dotenv

 
from typing import Union

##dodaje ale nie wiem czy to tu ma sens  

# Load .env file
load_dotenv()

client = openai.Client()   

frame_queue = queue.Queue()   
transcription_queue = queue.Queue()  


file_id = 0
text = ''
regex_pattern = r'\b(thanks?\s*(you\s*|for\s*)?|bye|hello).*'

###########################################
################### NAGRANIE 


''' 
# Basic parameters for recording
CHUNK = 1024  
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100   # u Bartka 16000
BUFFER_DURATION = 3  # Changed to 3 seconds
BASE_FILENAME = "output"
FILE_EXTENSION = ".wav"
OUTPUT_DIRECTORY = "AUDIOS"
BUFFER_MAX_SIZE = int(RATE / CHUNK * BUFFER_DURATION)

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

# Function to generate unique file name within the AUDIOS folder
def unique_filename(base, extension, directory):
    counter = 1
    while True:
        filename = os.path.join(directory, f"{base}_{counter}{extension}") if counter > 0 else os.path.join(directory, f"{base}{extension}")
        if not os.path.exists(filename):
            return filename
        counter += 1

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

# Initialize logging
logging.basicConfig(level=logging.INFO)


frames = []

# Record for 5 seconds
for i in range(0, int(RATE / CHUNK * BUFFER_DURATION)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

# Stop and close the stream
#stream.stop_stream()
#stream.close()
#p.terminate()

# Write data to a WAV file in the AUDIOS directory
WAVE_OUTPUT_FILENAME = unique_filename(BASE_FILENAME, FILE_EXTENSION, OUTPUT_DIRECTORY)
print(WAVE_OUTPUT_FILENAME) ##nazwa wygenerowanego pliku
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()



ASCII_IMG ="""                                
                 @@@@@                            
                 @@ @@                            
                   @                              
              @@@@@@@@@@@                 #       
           @@             @@          #     #     
        @@@   @@@@   @@@@   @@@   #    #     #     
      @  @   ,@  @   @  @   #@  #       #    #    
      @  @                  ,@  #      #     #    
         @    (@@@@@@@@@    ,@    #         #    
         @    @    @    @   @@        #    #      
          #@@@@@@@@@@@@@@@@@"""

''' 

#########################################################
#########################################################
 
'''''
def translate(file_name: str, temperature: float = 0.0, response_format: str = 'text', **kwargs) -> Union[str, dict]:
    """
    Translate the audio file with a timeout control.

    Args:
        file_name (str): Path to the audio file.
        temperature (float): Temperature for the translation model.
        response_format (str): Format of the response.
        **kwargs: Additional arguments for the translation model.

    Returns:
        Union[str, dict]: Translation result or empty string if timeout.
    """

    result = {"value": None}  # Use a dictionary to store the result

file_name  = "audios\output_40.wav"
    
def worker(file_name, **kwargs):
    with open(WAVE_OUTPUT_FILENAME, 'rb') as audio_data:
        result = {}
        result["value"] = client.audio.translations.create(
            model="whisper-1",
            file=audio_data, 
            temperature=0.0,
            response_format='text', 
            **kwargs)
        return result
    
    # Set up a thread to run the translation
translation_thread = threading.Thread(target=worker)
print(translation_thread)
translation_thread.start()
translation_thread.join(timeout=5)  # Timeout set to 5 seconds
'''

from openai import OpenAI
client = OpenAI()

#file_name  = "audios\output_38.wav"
file_name  = "audios\Rau_FYI_short.mp3"
#file_name  = "audios\magda_rau.m4a"

 
audio_file= open(file_name, "rb")
transcript_output = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file,
  response_format="text"
)
print(transcript_output)
 

 
 
#https://github.com/openai/whisper#available-models-and-languages 
#https://platform.openai.com/docs/guides/speech-to-text/improving-reliability
 
 
system_prompt = "Jestem RAU. Jeżdzę BMW. Robię RAP. Jestem z WWA. Elo WWR. Praga PŁD. Zwijam THC. W bletce OCB. Śmigam w RBK. Se po MOK. Na moim MTB. Nie znam BHP.Mam ADHD .Jem GMO. Wierzę w UFO.Polskie DNA. Nie gram w GTA.  Słucham LSO. Zerkam w HBO."

#system_prompt = "Jestem RAU. Jeżdzę BMW. Robię RAP. Jestem z WWA. Elo WWR. Praga PŁD."
print(system_prompt)

'''
with open('rau_transcribe.txt', 'r',  encoding='utf-8') as file:
    transcribe = file.read()
print(transcribe)
''' 

#def transcribe(file_name, prompt=system_prompt)
 
def transcribe(file_name, system_prompt):
    # Your code to transcribe the audio file goes here
    # This is just a placeholder
    transcribed_text = " "
    return transcribed_text



system_instruction = "Update transcribed words, by using the provided dictionary. Do not change transcribed language, do not translate text"

def generate_corrected_transcript(temperature, system_prompt, file_name):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo-0125", 
        #model="gpt-3.5-turbo-16k", legacy
#        model = "gpt-4-0613",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_instruction + system_prompt
            },
            {
                "role": "user",
                "content": transcribe(file_name, system_prompt)
            }
        ]
    )
    #return response['choices'][0]['message']['content']
    return response

corrected_text = generate_corrected_transcript(0, system_instruction + system_prompt, transcribe)
print(corrected_text) 
 