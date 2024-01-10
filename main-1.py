import logging
import os
import queue
import re
import threading
import time
import tkinter as tk
import wave
from typing import Union
from time import sleep
import librosa
import numpy as np
import openai
import pyaudio
from PIL import Image, ImageTk
from dotenv import load_dotenv

load_dotenv()
client = openai.Client()

frame_queue = queue.Queue()
transcription_queue = queue.Queue()
file_queue = queue.Queue()

frames = []
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

ROOT_PATH='audios'
os.makedirs(ROOT_PATH, exist_ok=True)

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


def is_speech_present(audio_file: str, threshold: float = 0.02) -> bool:
    """
    Check if speech is present in the audio file.

    Args:
        audio_file (str): Path to the audio file.
        threshold (float): Threshold for determining if speech is present.

    Returns:
        bool: True if speech is present, False otherwise.
    """
    y, sr = librosa.load(audio_file)

    abs_y = np.abs(y)
    max_sound_level = np.max(abs_y)
    if max_sound_level > threshold:
        return True
    else:
        logging.info("No significant sound detected")
        return False


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

    result = {"value": None}
    logging.info("translate")

    def worker():
        try:
            with open(file_name, 'rb') as audio_data:
                result["value"] = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_data, temperature=temperature,
                    response_format=response_format, **kwargs)
        except Exception as e:
            logging.error(f"Error in translation worker: {e}")

    translation_thread = threading.Thread(target=worker)
    translation_thread.start()
    translation_thread.join(timeout=5)  # Timeout set to 5 seconds

    if translation_thread.is_alive():
        logging.warning("Translation thread timed out")
        return ""
    elif result["value"] is None:
        logging.error("Translation failed or returned no result")
        return ""
    else:
        return result["value"]


def transcript(file_name: str, temperature: float = 0.0, response_format: str = 'text', **kwargs) -> Union[str, dict]:
    """
    Transcribe the audio file.

    Args:
        file_name (str): Path to the audio file.
        temperature (float): Temperature for the transcription model.
        response_format (str): Format of the response.
        **kwargs: Additional arguments for the transcription model.

    Returns:
        Union[str, dict]: Transcription result.
    """
    try:
        with open(file_name, 'rb') as audio_data:
            return client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_data, temperature=temperature, response_format=response_format, **kwargs)
    except Exception as e:
        logging.error(f"Error in transcript: {e}")
        return {"error": str(e)}


def translate_with_whisper():
    logging.info("translate_with_whisper")
    while True:
        try:
            file_name = os.path.join('translator_app_with_memory', 'audios', sorted(os.listdir('../audios'))[0])
            if file_name and is_speech_present(file_name):
                if is_speech_present(file_name):
                    response = translate(file_name)
                    logging.info(f'Response: {response}')

                    if (re.search(regex_pattern, response, re.IGNORECASE) and len(response) < 15) or len(response) < 2:
                        logging.info(f'Filter out')
                        response = ''
                else:
                    response = ''
                transcription_queue.put(response)
                os.remove(file_name)
                logging.info(f"File {file_name} removed.")
        except Exception as e:
            logging.error(f"Error in translate_with_whisper: {e}")


def guard_with_llm(conversation=conversation):
    """
    Handle the translation of transcribed text.
    """
    logging.info("guarding")
    while True:
        try:
            if not transcription_queue.empty():
                return conversation.run(transcription_queue.get())
            sleep(0.1)  # Prevents high CPU usage in idle state
        except Exception as e:
            logging.error(f"Error in guard_with_llm: {e}")
            sleep(0.1)  # To prevent rapid log flooding


def _generate_file_name(file_id: int) -> str:
    """
    Generate a file name based on the file_id.

    Args:
        file_id (int): The file ID.

    Returns:
        str: The generated file name.
    """
    return f'audios/output_{file_id}.wav'


def write_audio(file_id: int = file_id) -> None:
    """
    Write audio data to a file and translate it.

    Args:
        file_id (int): The file ID.
        text (str): The text to be translated.
    """
    logging.info('Writing audio')
    while True:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frames.append(frame)
                if len(frames) == BUFFER_MAX_SIZE:
                    os.makedirs('../audios', exist_ok=True)
                    file_name = _generate_file_name(file_id)
                    file_id += 1
                    with wave.open(file_name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                    frames.clear()
        except Exception as e:
            logging.error(f"Error in write_audio: {e}")


def record_audio() -> None:
    """
    Record audio data and put it into the frame queue.
    """
    try:
        while True:
            logging.info('Recording audio')
            data = stream.read(CHUNK, exception_on_overflow=False)
            frame_queue.put(data)
            if frame_queue.qsize() >= BUFFER_MAX_SIZE:
                break
    except Exception as e:
        logging.error(f"Error in record_audio: {e}")
    finally:
        logging.info("Audio recording stopped")


def _sort_and_remove_first(directory):
    """
    Sorts a list of file names alphabetically and removes the first file in the sorted list.

    :param files: A list of file names (strings).
    :return: The sorted list of files with the first file removed.
    """
    try:
        sorted_files = sorted(os.listdir(directory))
        if sorted_files:
            return os.path.join(directory, sorted_files[0])
    except Exception as e:
        logging.error(f"Error in _sort_and_remove_first: {e}")
        return None



class TranscriptionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Subtitle.me")
        # Define colors and fonts
        backgroundColor = '#000000'
        primaryTextColor = '#ffffff'
        secondaryTextColor = '#dcdcdc'
        fontLargeBold = ('Arial', 18, 'bold')
        fontMedium = ('Arial', 12)

        self.root.configure(bg=backgroundColor)

        screen_width = self.root.winfo_screenwidth()
        app_height = 200  # Adjust the height as needed

        self.original_logo_image = Image.open(os.path.join(os.path.dirname(__file__), 'translator_app_with_memory', 'images', 'logo_white.png'))
        self.original_clock_image = Image.open(os.path.join(os.path.dirname(__file__), 'translator_app_with_memory', 'images', 'clock.png'))

        # Set window attributes
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.9)
        self.root.geometry(f"{screen_width}x{app_height}")

        # Create frames for layout with the specified width ratios
        left_frame = tk.Frame(self.root, bg=backgroundColor, width=int(screen_width * 0.2))
        center_frame = tk.Frame(self.root, bg=backgroundColor, width=int(screen_width * 0.6))
        right_frame = tk.Frame(self.root, bg=backgroundColor, width=int(screen_width * 0.2))

        self.logo_photo_image = self.rescale_image(self.original_logo_image, 200, 200)
        self.clock_photo_image = self.rescale_image(self.original_clock_image, 50,50)
        self.logo_label = tk.Label(left_frame, image=self.logo_photo_image, bg=backgroundColor)
        self.logo_label.pack(side='top', pady=10)

        self.clock_label = tk.Label(right_frame, image=self.clock_photo_image, bg=backgroundColor)
        self.clock_label.pack(side='top', pady=0)

        self.previous_var = tk.StringVar()
        self.previous_label = tk.Label(center_frame, textvariable=self.previous_var, fg=secondaryTextColor,
                                       bg=backgroundColor, font=fontMedium, wraplength=center_frame.winfo_reqwidth())
        self.previous_var.set("")
        self.previous_label.pack(pady=(10, 50), side='bottom')

        self.current_var = tk.StringVar()
        self.current_label = tk.Label(center_frame, textvariable=self.current_var, fg=primaryTextColor,
                                      bg=backgroundColor, font=fontLargeBold,  wraplength=center_frame.winfo_reqwidth())
        self.current_var.set("")
        self.current_label.pack(pady=(50, 10), side='top',)

        delay_label = tk.Label(right_frame, text="~5 second delay may occur", fg=primaryTextColor, bg=backgroundColor,
                               font=fontMedium)
        delay_label.pack(side='top', pady=5)

        self.root.bind('<Configure>', self._on_resize)

        # Pack the frames into the root window
        left_frame.pack(side='left', fill='y')
        center_frame.pack(side='left', fill='both', expand=True)
        right_frame.pack(side='left', fill='y')

        left_frame.pack_propagate(False)
        center_frame.pack_propagate(False)
        right_frame.pack_propagate(False)

        self.update_thread = threading.Thread(target=self.update_text)
        self.update_thread.daemon = True
        self.update_thread.start()

    def update_text(self) -> None:
        """
        Update the text in the transcription app.
        """
        try:
            while True:
                if not transcription_queue.empty():
                    transcription = transcription_queue.get().split('\n')
                    logging.info(f'transcription: {transcription}')
                    if len(transcription) > 0 and any(string.strip() for string in transcription):
                        self.current_var.set(transcription[0])
                        if len(transcription) > 1:
                            self.previous_var.set(transcription[1])

                    else:
                        self.current_var.set('...')

                time.sleep(1)
        except Exception as e:
            logging.error(f"Error in update_text: {e}")
        finally:
            self.root.after(1000, self.update_text)

    def _on_resize(self, event):
        # Rescale the logo image
        new_logo_image = self.rescale_image(self.original_logo_image, 150, 150)  # Adjust dimensions as needed
        self.logo_label.config(image=new_logo_image)
        self.logo_label.image = new_logo_image

        # Rescale the clock image
        new_clock_image = self.rescale_image(self.original_clock_image, 150, 150)  # Adjust dimensions as needed
        self.clock_label.config(image=new_clock_image)
        self.clock_label.image = new_clock_image


    def rescale_image(self, original_image, width, height):
        resized_image = original_image.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized_image)

if __name__ == '__main__':
    root = tk.Tk()
    app = TranscriptionApp(root)

    recording_thread = threading.Thread(target=record_audio)
    translate_thread = threading.Thread(target=translate_with_whisper)
    guard_thread = threading.Thread(target=guard_with_llm)
    writing_thread = threading.Thread(target=write_audio)


    recording_thread.start()
    guard_thread.start()
    translate_thread.start()
    writing_thread.start()

    root.mainloop()

    recording_thread.join()
    guard_thread.join()
    translate_thread.join()
    writing_thread.join()


    stream.stop_stream()
    stream.close()
    p.terminate()

