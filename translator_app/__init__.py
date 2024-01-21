import os

import pyaudio
import queue
import logging
from dotenv import load_dotenv
import openai

load_dotenv()
client = openai.Client()

frame_queue = queue.Queue()
transcription_queue = queue.Queue()
printout_queue = queue.Queue()


# Audio configuration
RATE = 16000  # Sample rate
CHUNK = 1024  # Chunk size
BUFFER_DURATION = 5  # Buffer duration in seconds
BUFFER_MAX_SIZE = int(RATE / CHUNK * BUFFER_DURATION)

file_id = 0
text = ""
regex_pattern = r"\b(thanks?\s*(you\s*|for\s*)?|bye|hello).*"

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=3,
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

#
ROOT_PATH = "audios"
os.makedirs(ROOT_PATH, exist_ok=True)

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny.en"
)
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)


ASCII_IMG = """                                
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
