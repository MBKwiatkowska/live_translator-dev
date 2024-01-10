import logging
import os
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
from . import moderated_queue


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

        self.original_logo_image = Image.open(os.path.join(os.path.dirname(__file__), 'images', 'logo_white.png'))
        self.original_clock_image = Image.open(os.path.join(os.path.dirname(__file__), 'images', 'clock.png'))

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
                if not moderated_queue.empty():
                    moderated_transcription = moderated_queue.get().split('\n')
                    logging.info(f'moderated transcription: {moderated_transcription}')
                    if len(moderated_transcription) > 0 and any(string.strip() for string in moderated_transcription):
                        self.current_var.set(moderated_transcription[0])
                        if len(moderated_transcription) > 1:
                            self.previous_var.set(moderated_transcription[1])

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