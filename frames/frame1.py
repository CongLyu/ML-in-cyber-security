import tkinter as tk
from tkinter import filedialog
from data_processing.data_loader import load_dataset


class Frame1(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Use 'expand' to allow the widget to expand to fill any space not otherwise used
        # Use 'fill' to fill the space in the x and y directions
        self.pack(expand=True, fill='both')

        # Increase the button size to make the frame larger, you can adjust the 'padx' and 'pady' as needed
        load_data_button = tk.Button(self, text="Load Data", command=self.load_data, padx=50, pady=20)
        load_data_button.pack(pady=20)  # Add padding to push the frame's boundaries out

        next_frame_button = tk.Button(self, text="Next Frame", command=lambda: controller.show_frame("Frame2"), padx=50, pady=20)
        next_frame_button.pack(pady=20)

    def load_data(self):
        # Ask the user for the location of the XML file
        file_path = filedialog.askopenfilename(filetypes=[("XML files", "*.xml")])
        if file_path:
            # Clear previous shared data
            self.controller.reset_shared_data()
            # Load the new dataset
            dataset = load_dataset(file_path)
            # Update the shared data with the new dataset
            self.controller.update_shared_data('dataset', dataset)
            # Notify that the data has been loaded
            self.controller.data_loaded()  # Implement this method to update the UI as needed

