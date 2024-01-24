from asyncio.log import logger
import os
import re
import time
import threading
import wave
import pyaudio
import librosa
import numpy as np
from typing import Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
from . import (
    client,
    stream,
    frame_queue,
    p,
    printout_queue,
    transcription_queue,
    translation_queue,
    BUFFER_MAX_SIZE,
    RATE,
    CHUNK,
    TRANSLATION_SYSTEM_MESSAGE,
)


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
    logging.info("program set up. You can start speaking")
    abs_y = np.abs(y)
    max_sound_level = np.max(abs_y)
    if max_sound_level > threshold:
        return True
    else:
        logging.info("No significant sound detected")
        return False


def transcript(
    temperature: float = 0.0,
    response_format: str = "text",
    **kwargs,
) -> None:
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
    while True:
        if not transcription_queue.empty():
            file_name = transcription_queue.get()
            # if is_speech_present(file_name):
            with open(file_name, "rb") as audio_data:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_data,
                    temperature=temperature,
                    response_format=response_format,
                    # language='pl',
                    **kwargs,
                )
            logging.info(f"Response: {response}")

            translation_queue.put(response)
            os.remove(file_name)
            logging.info(f"transcription of {file_name} finished!")


def translate(
    temperature: float = 0.0001,
    **kwargs,
) -> Union[str, dict]:
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
    while True:
        if not translation_queue.empty():
            message_to_translate = translation_queue.get()
            prepared_input = [
                {"role": "system", "content": TRANSLATION_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": message_to_translate,
                },
            ]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prepared_input,
                temperature=temperature,
                **kwargs,
            )
            translation = response.choices[0].message.content
            printout_queue.put(translation)
            logging.info(f"created_translation: {translation}")


def _generate_file_name(file_id: int) -> str:
    """
    Generate a file name based on the file_id.

    Args:
        file_id (int): The file ID.

    Returns:
        str: The generated file name.
    """
    return f"audios/output_{file_id}.wav"


def write_audio(file_id: int = 0) -> None:
    """
    Write audio data to a file and translate it.

    Args:
        file_id (int): The file ID.
        text (str): The text to be translated.
    """
    frames = []
    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frames.append(frame)
                if len(frames) == BUFFER_MAX_SIZE:
                    os.makedirs("../audios", exist_ok=True)
                    file_name = _generate_file_name(file_id)
                    file_id += 1
                    with wave.open(file_name, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(RATE)
                        wf.writeframes(b"".join(frames))
                    logging.info(f"adding to queue {file_name}")
                    transcription_queue.put(file_name)
                    frames = []
    except Exception as e:
        logging.error(f"Error in write_audio: {e}")
    finally:
        if "wf" in locals():
            wf.close()


def record_audio() -> None:
    """
    Record audio data and put it into the frame queue.
    """
    while True:
        try:
            data = stream.read(CHUNK)
            frame_queue.put(data)
            if frame_queue.qsize() >= BUFFER_MAX_SIZE:
                print("break condition met")
                break
        except Exception as e:
            print("recording error")
            logging.error(f"Error in record_audio: {e}")
