import os
import threading
import wave
from time import sleep

import pyaudio
import librosa
import numpy as np
from typing import Union
import logging

from . import (client, stream, frame_queue, transcription_queue, p, BUFFER_MAX_SIZE, RATE, CHUNK, regex_pattern,
               conversation, file_id, AUDIO_PATH, moderated_queue)



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
        input()
        return {"error": str(e)}


def translate_with_whisper():
    while True:
        try:
            sorted_files = sorted(os.listdir(AUDIO_PATH))
            if sorted_files:

                file_name = os.path.join(AUDIO_PATH, sorted_files[0])
                if file_name:
                    if is_speech_present(file_name):
                        response = translate(file_name)
                        logging.info(f'Response: {response}')
                    else:
                        response = ''
                    transcription_queue.put(response)
                    logging.error(f'Whisper {response}')
                    os.remove(file_name)
                    logging.info(f"File {file_name} removed.")

        except Exception as e:
            logging.error(f"Error in translate_with_whisper: {e}")
            input()
            pass


def guard_with_llm(conversation=conversation):
    """
    Handle the translation of transcribed text.
    """
    logging.info("guarding")
    while True:
        try:
            if not transcription_queue.empty():
                logging.error(transcription_queue.get())
                moderated = conversation.run(transcription_queue.get())
                moderated_queue.put(moderated.replace('\n', '').replace('\t', '').replace('\r', ''))
        except Exception as e:
            logging.error(f"Error in guard_with_llm: {e}")
            input()
            pass



def _generate_file_name(file_id: int) -> str:
    """
    Generate a file name based on the file_id.

    Args:
        file_id (int): The file ID.

    Returns:
        str: The generated file name.
    """
    return os.path.join(AUDIO_PATH, f'output_{file_id}.wav')


def write_audio(file_id: int = file_id) -> None:
    """
    Write audio data to a file and translate it.

    Args:
        file_id (int): The file ID.
        text (str): The text to be translated.
    """
    logging.info('Writing audio')
    frames = []
    while True:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frames.append(frame)
                if len(frames) >= BUFFER_MAX_SIZE:
                    os.makedirs(AUDIO_PATH, exist_ok=True)
                    file_name = _generate_file_name(file_id)
                    file_id += 1
                    with wave.open(file_name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                    frames = frames[-int(0.2 * len(frames)):]
        except Exception as e:
            logging.error(f"Error in write_audio: {e}")
            pass


def record_audio() -> None:
    """
    Record audio data and put it into the frame queue.
    """
    try:
        while True:
            logging.info('Recording audio')
            data = stream.read(CHUNK, exception_on_overflow=False)
            frame_queue.put(data)
            if frame_queue.qsize() > 2* BUFFER_MAX_SIZE:
                sleep(5)
    except Exception as e:
        logging.error(f"Error in record_audio: {e}")
        pass



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
