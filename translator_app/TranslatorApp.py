import os
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk

import translator_app.globals as globals

from translator_app.__init__ import printout_queue, logging


class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("App name")

        # Define colors and fonts
        backgroundColor = "#000000"
        primaryTextColor = "#ffffff"
        secondaryTextColor = "#dcdcdc"
        fontLargeBold = ("Arial", 18, "bold")
        fontMedium = ("Arial", 12)
        self.text_for_previous_message = (
            "Application is starting. Please, wait..."
        )

        self.root.configure(bg=backgroundColor)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        app_height = 200

        self.original_logo_image = Image.open(
            os.path.join(os.path.dirname(__file__), "images", "logo_white.png")
        )
        self.original_clock_image = Image.open(
            os.path.join(os.path.dirname(__file__), "images", "clock.png")
        )
        position_x = int((screen_width / 2) - (screen_width / 2))
        position_y = int(screen_height - app_height)
        # Set window attributes
        self.root.attributes("-topmost", True)# on top
        self.root.attributes("-alpha", 0.9)# window transparency
        self.root.geometry(f"{screen_width}x{app_height}+{position_x}+{position_y}")

        # Create frames for layout with the specified width ratios
        left_frame = tk.Frame(
            self.root, bg=backgroundColor, width=int(screen_width * 0.2)
        )
        center_frame = tk.Frame(
            self.root, bg=backgroundColor, width=int(screen_width * 0.6)
        )
        right_frame = tk.Frame(
            self.root, bg=backgroundColor, width=int(screen_width * 0.2)
        )

        self.logo_photo_image = self.rescale_image(
            self.original_logo_image, 200, 200
        )

        self.clock_photo_image = self.rescale_image(
            self.original_clock_image, 50, 50
        )
        self.logo_label = tk.Label(
            left_frame, image=self.logo_photo_image, bg=backgroundColor
        )
        self.logo_label.pack(side="top", pady=10)

        self.clock_label = tk.Label(
            right_frame, image=self.clock_photo_image, bg=backgroundColor
        )
        self.clock_label.pack(side="top", pady=0)

        self.previous_var = tk.StringVar()
        self.previous_label = tk.Label(
            center_frame,
            textvariable=self.previous_var,
            fg=secondaryTextColor,
            bg=backgroundColor,
            font=fontMedium,
            wraplength=center_frame.winfo_reqwidth(),
        )
        self.previous_var.set("")
        self.previous_label.pack(pady=(10, 50), side="bottom")

        self.current_var = tk.StringVar()
        self.current_label = tk.Label(
            center_frame,
            textvariable=self.current_var,
            fg=primaryTextColor,
            bg=backgroundColor,
            font=fontLargeBold,
            wraplength=center_frame.winfo_reqwidth(),
        )
        self.current_var.set(self.text_for_previous_message)
        self.current_label.pack(
            pady=(50, 10),
            side="top",
        )
        print("define delay label")
        delay_label = tk.Label(
            right_frame,
            text="~5 second delay may occur",
            fg=primaryTextColor,
            bg=backgroundColor,
            font=fontMedium,
        )
        print("call delay label")
        delay_label.pack(side="top", pady=5)
        print("call on_resize")
        self.root.bind("<Configure>", self._on_resize)

        # Pack the frames into the root window
        left_frame.pack(side="left", fill="y")
        center_frame.pack(side="left", fill="both", expand=True)
        right_frame.pack(side="left", fill="y")
        root.protocol("WM_DELETE_WINDOW", self._on_closing)
        left_frame.pack_propagate(False)
        center_frame.pack_propagate(False)
        right_frame.pack_propagate(False)
        print("add_update_text_to_threading")
        #        logging.info("starting thread update")
        self.update_thread = threading.Thread(target=self.update_text)
        print("set update thread daemon to true")
        #        logging.info("set update_thread daemon to true")
        self.update_thread.daemon = True
        print("start thread")
        self.update_thread.start()

    #        logging.info("started update thread")

    def update_text(self) -> None:
        """
        Update the text in the transcription app.
        """
        while True:
            if globals.run_threads:
                if not printout_queue.empty():
                    self.previous_var.set(self.text_for_previous_message)
                    transcription = printout_queue.get()
                    #                    logging.info(f"received transcription {transcription}")
                    if len(transcription) > 0 and any(
                        string.strip() for string in transcription
                    ):
                        self.current_var.set(transcription)
                        self.text_for_previous_message = transcription
                    #                        logging.info(f"printed transcription {transcription}")
                    else:
                        self.current_var.set("...")
                time.sleep(1)
            else:
                break

    def _on_resize(self, event):
        # Rescale the logo image
        new_logo_image = self.rescale_image(
            self.original_logo_image, 150, 150
        )  # Adjust dimensions as needed
        self.logo_label.config(image=new_logo_image)
        self.logo_label.image = new_logo_image

        # Rescale the clock image
        new_clock_image = self.rescale_image(
            self.original_clock_image, 150, 150
        )  # Adjust dimensions as needed
        self.clock_label.config(image=new_clock_image)
        self.clock_label.image = new_clock_image

    def rescale_image(self, original_image, width, height):
        resized_image = original_image.resize(
            (width, height), Image.Resampling.LANCZOS
        )
        return ImageTk.PhotoImage(resized_image)

    def _on_closing(self):
        logging.info("triggered on closing function")
        globals.run_threads = (
            False  # Set the flag to False to stop the threads
        )
        logging.info(f"run_threads_on_close = {globals.run_threads}")
        self.root.destroy()
