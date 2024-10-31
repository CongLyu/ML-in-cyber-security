import tkinter as tk
from controller import ApplicationController
from frames.frame1 import Frame1
from frames.frame2 import Frame2
from frames.frame3 import Frame3
from frames.frame4 import Frame4


class MainApplication(tk.Tk):
    def __init__(self, *args, **kwargs):
        global result_frame_three, result_frame_two
        super().__init__(*args, **kwargs)

        self.shared_data = {
            'dataset': None,
            'dataset_reduced': None,
            'dataset_covariates': None,
            'dataset_covariates_encoded': None,
            'X_train': None,
            'X_test': None,
            'dropped_columns': [],
            'target_column': None,
            'dataset_response': None,
            'dataset_response_encoded': None,
            'Y_train': None,
            'Y_test': None,
            'Y_predicted': {},
            'selected_model': None,
            'test_size': None,
            'models': {},
            'evaluation_metrics': None,
            'label_encoder': {},
        }

        self.new_data = {
            'dataset': None,
            'dataset_covariates': None,
            'dataset_covariates_encoded': None,
            'dataset_response': None,
            'dataset_response_encoded': None,
        }

        self.controller = ApplicationController(self)  # Create a controller instance and pass the main app to it

        # This container will use pack for its children
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)

        self.frames = {}
        for F in (Frame1, Frame2, Frame3, Frame4):
            frame_name = F.__name__
            frame = F(parent=self.container, controller=self.controller)
            self.frames[frame_name] = frame
            # Pack is used here, but we keep the frames hidden by default
            frame.pack(fill="both", expand=True)
            frame.pack_forget()

        self.show_frame("Frame1")

    def show_frame(self, frame_name):
        # Hide all frames
        for frame in self.frames.values():
            frame.pack_forget()

        if frame_name == 'Frame1':
            # Show the requested frame
            frame = self.frames[frame_name]
            frame.pack(fill="both", expand=True)

        if frame_name == 'Frame2':
            # Show the requested frame
            frame = self.frames[frame_name]
            frame.update_display()
            frame.pack(fill="both", expand=True)

        if frame_name == 'Frame3':
            frame = self.frames[frame_name]
            frame.pack(fill="both", expand=True)

        if frame_name == 'Frame4':
            frame = self.frames[frame_name]
            frame.pack(fill="both", expand=True)

    def update_shared_data(self, key, value):
        self.shared_data[key] = value

    def get_shared_data(self, key):
        return self.shared_data.get(key)

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
