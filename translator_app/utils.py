import io
import os
import requests
import soundfile as sf
import wave
import pyaudio
import librosa
import numpy as np
import logging

from typing import Union

import translator_app.globals as globals

from translator_app.__init__ import *


def cleanup_audios():
    files = os.listdir("audios")
    for file in files:
        try:
            os.remove(f"audios/{file}")
        except:
            logging.info(f"failed to clean up {file}")


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
        if globals.run_threads:
            if not transcription_queue.empty():
                file_name = transcription_queue.get()
                # if is_speech_present(file_name):
                if AUDIO_MODEL == "openai":
                    response = transcript_with_openai(
                        file_name=file_name,
                        temperature=temperature,
                        response_format=response_format,
                    )
                elif AUDIO_MODEL == "scalepoint":
                    response = transcript_with_scalepoint(
                        file_name=file_name,
                    )
                elif AUDIO_MODEL == "scalepoint_translation":
                    response = translate_with_scalepoint(
                        file_name=file_name,
                    )
                elif AUDIO_MODEL == "faster-whisper":
                    response = transcript_with_faster_whisper(
                        file_name=file_name
                    )
                elif AUDIO_MODEL == "google-cloud-speech":
                    response = transcript_with_google_cloud_speech(
                        file_name=file_name
                    )
                else:
                    response = transcript_with_hugging_face_whisper(
                        file_name=file_name
                    )

                if file_name == "audios/output_0.wav":
                    os.remove(file_name)
                    while transcription_queue.qsize() > 1:
                        temp_file = transcription_queue.get()
                        os.remove(temp_file)
                        logging.info(
                            f"removing {temp_file} - too long in a queue"
                        )
                else:
                    while transcription_queue.qsize() > 2:
                        temp_file = transcription_queue.get()
                        os.remove(temp_file)
                        logging.info(
                            f"removing {temp_file} - too long in transcription queue"
                        )
                    if len(response) > 0:
                        logging.info(f"transcription: {response}")
                        if AUDIO_MODEL == "scalepoint_translation":
                            printout_queue.put(response)
                        else:
                            translation_queue.put(response)
                    os.remove(file_name)
        else:
            break


def transcript_with_google_cloud_speech(
    file_name: str,
    **kwargs,
):
    with open(file_name, "rb") as f:
        content = f.read()
    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{google_speech_project_id}/locations/global/recognizers/_",
        config=google_speech_config,
        content=content,
    )
    response = google_speech_client.recognize(request=request)
    try:
        result = response.results[0].alternatives[0].transcript
        logging.info(f"Google transcription for {file_name} succeeded.")
    except:
        logging.info(
            f"Google transcription for {file_name} failed. \n"
            + f"Sent request: {request.config} {request.config_mask} {request.recognizer} {request.uri}.\n"
            + f"Response body: {response}."
        )
        result = ""
    return result


def transcript_with_openai(
    file_name: str,
    temperature: float,
    response_format: str,
    **kwargs,
):
    with open(file_name, "rb") as audio_data:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_data,
            temperature=temperature,
            response_format=response_format,
            language=INPUT_LANGUAGE,
            prompt=OPENAI_WHISPER_PROMPT,
            **kwargs,
        )
    return response


def transcript_with_scalepoint(
    file_name: str,
):

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {SCALEPOINT_BEARER}",
    }
    with open(file_name, "rb") as f:
        wav_content = f.read()
        wav_io = io.BytesIO(wav_content)
        files = {"file": ("audio.wav", wav_io, "audio/wav")}
    response = requests.post(
        f"{SCALEPOINT_ENDPOINT}/transcriptions/?response_format=text",
        headers=headers,
        files=files,
    )
    return response.json()["text"]


def transcript_with_faster_whisper(file_name: str):
    result: str = ""
    segments, _ = model.transcribe(file_name)
    segments = list(segments)
    for segment in segments:
        if segment.no_speech_prob < 0.6:
            result = result + segment.text
    return result


def transcript_with_hugging_face_whisper(file_name: str):
    audio, sampling_rate = sf.read(file_name)
    input_features = processor(
        audio, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features
    predicted_ids = model.generate(input_features)
    response = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
        0
    ]
    return response


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
            while transcription_queue.qsize() > 2:
                temp_file = translation_queue.get()
                logging.info(
                    f"removing {temp_file} - too long in transcription queue"
                )
            message_to_translate = translation_queue.get()
            prepared_input = [
                {"role": "system", "content": TRANSLATION_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": message_to_translate,
                },
            ]

            ################## translate with openai
            response = client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=prepared_input,
                temperature=temperature,
                **kwargs,
            )
            translation = response.choices[0].message.content
            logging.info(f"translation: {translation}")
            while printout_queue.qsize() > 2:
                temp_file = printout_queue.get()
                logging.info(
                    f"removing {temp_file} - too long in printout queue"
                )
            printout_queue.put(translation)


def translate_with_scalepoint(
    file_name: str,
):

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {SCALEPOINT_BEARER}",
    }
    with open(file_name, "rb") as f:
        wav_content = f.read()
        wav_io = io.BytesIO(wav_content)
        files = {"file": ("audio.wav", wav_io, "audio/wav")}
    response = requests.post(
        f"{SCALEPOINT_ENDPOINT}/translations/?response_format=text&language=polish",
        headers=headers,
        files=files,
    )
    result = response.json()["text"]
    logging.info(f"translation: {result}")
    return result


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
            if globals.run_threads:
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
                        #                        logging.info(f"adding to queue {file_name}")
                        transcription_queue.put(file_name)
                        frames = []

            else:
                break
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
        if globals.run_threads:
            try:
                data = stream.read(CHUNK)
                frame_queue.put(data)
                if frame_queue.qsize() >= BUFFER_MAX_SIZE:
                    print("break condition met")
                    break
            except Exception as e:
                print("recording error")
                logging.error(f"Error in record_audio: {e}")
        else:
            break
