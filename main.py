import logging
import threading
import tkinter as tk

from translator_app import stream, p
from translator_app.TranslatorApp import TranscriptionApp
from translator_app.utils import (
    cleanup_audios,
    write_audio,
    record_audio,
    transcript,
    translate,
)

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    recording_thread = threading.Thread(target=record_audio)
    writing_thread = threading.Thread(target=write_audio)
    transcript_thread = threading.Thread(target=transcript)
    translate_thread = threading.Thread(target=translate)

    recording_thread.start()
    writing_thread.start()
    logging.info("start transcript thread")
    transcript_thread.start()
    logging.info("start translate thread")
    translate_thread.start()
    root.mainloop()
    logging.info("join recording thread")
    recording_thread.join()
    logging.info("join writing thread")
    writing_thread.join()
    logging.info("stopping stream")
    stream.stop_stream()
    logging.info("closing stream")
    stream.close()
    logging.info("terminating app")
    p.terminate()
    cleanup_audios()
