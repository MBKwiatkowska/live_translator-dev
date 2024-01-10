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
moderated_queue = queue.Queue()
file_queue = queue.Queue()

# Audio configuration
RATE = 16000  # Sample rate
CHUNK = 1024  # Chunk size
BUFFER_DURATION = 5  # Buffer duration in seconds
BUFFER_MAX_SIZE = int(RATE / CHUNK * BUFFER_DURATION)

file_id = 0
text = ''
regex_pattern = r'\b(thanks?\s*(you\s*|for\s*)?|bye|hello).*'

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize logging
logging.basicConfig(level=logging.INFO)

AUDIO_PATH = os.path.join('translator_app_with_memory', 'audios')
os.makedirs(AUDIO_PATH, exist_ok=True)
logging.info(os.path.abspath(AUDIO_PATH))


from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import (
    CombinedMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
)
from langchain.prompts import PromptTemplate

load_dotenv()

_DEFAULT_TEMPLATE = """
    Prompt: "Given the following excerpt from a presentation speech, generate the corresponding text that would appear in a PowerPoint slide or document related to the speech content. If the speech includes factual content, the text should accurately reflect this information in a concise, bullet-point format. However, if the speech contains fabrications or hallucinations, as indicated by the transcriber, return an empty string."

    Summary of presentation:
    {history}
    Current speech history (last 3 sentences):
    {chat_history_lines}
    Current sentence: {input}
    AI:
    """

conv_memory = ConversationBufferWindowMemory(k=3,
                                             memory_key="chat_history_lines",
                                             input_key="input"
                                             )
prompt = PromptTemplate(input_variables=["history", "input", "chat_history_lines"],
                        template=_DEFAULT_TEMPLATE)

summary_memory = ConversationSummaryMemory(llm=OpenAI(), input_key="input")

memory = CombinedMemory(memories=[conv_memory, summary_memory])

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=False, memory=memory, prompt=prompt)




