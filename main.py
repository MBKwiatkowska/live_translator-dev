import logging
import threading
import tkinter as tk

# from translator_app_with_memory import stream, p
# from translator_app_with_memory.TranslatorApp import TranscriptionApp
# from translator_app_with_memory.utils import write_audio, record_audio, translate_with_whisper, guard_with_llm

from translator_app import stream, p
from translator_app.TranslatorApp import TranscriptionApp
from translator_app.utils import write_audio, record_audio, transcript

#
# if __name__ == '__main__':
#     root = tk.Tk()
#     app = TranscriptionApp(root)
#
#     recording_thread = threading.Thread(target=record_audio)
#     translate_thread = threading.Thread(target=translate_with_whisper)
#     guard_thread = threading.Thread(target=guard_with_llm)
#     writing_thread = threading.Thread(target=write_audio)
#
#
#     recording_thread.start()
#     translate_thread.start()
#     guard_thread.start()
#     writing_thread.start()
#
#     root.mainloop()
#
#     recording_thread.join()
#     translate_thread.join()
#     guard_thread.join()
#     writing_thread.join()
#
#
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
# #
# #

# from translator_app_test import stream, p
# from translator_app_test.TranslatorApp import TranscriptionApp
# from translator_app_test.utils import write_audio, record_audio

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    recording_thread = threading.Thread(target=record_audio)
    writing_thread = threading.Thread(target=write_audio, args=(0, ""))
    transcript_thread = threading.Thread(target=transcript)

    recording_thread.start()
    writing_thread.start()
    logging.info("start transcript thread")
    transcript_thread.start()
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
