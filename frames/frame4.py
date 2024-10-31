import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from data_processing.data_loader import load_dataset
import seaborn as sns


class Frame4(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Create a button to go back to Frame 3
        back_button = tk.Button(self, text="Back to Frame 3", command=lambda: controller.show_frame("Frame3"))
        back_button.pack()

        # Add attributes to keep track of the plot widgets
        self.cm_frames = []  # List to keep track of confusion matrix frames
        self.roc_canvas_widget = None  # To keep track of the ROC curve canvas widget

        # Create a button that will trigger the model evaluation process
        evaluate_model_button = tk.Button(self, text="Evaluate Model", command=self.evaluate_model)
        evaluate_model_button.pack()

        # Button for loading XML data
        load_button = tk.Button(self, text="Load XML Data", command=self.load_xml_data)
        load_button.pack()

        # Create a button to trigger the evaluation
        evaluate_button = tk.Button(self, text="Predict new dataset", command=self.upload_and_evaluate_data)
        evaluate_button.pack()

        # Attributes to keep track of plot widgets and accuracy label
        self.accuracy_label_widget = None
        self.cm_canvas_widget = None  # Keep track of the confusion matrix widget



    def clear_previous_plots(self):
        # Clear previous confusion matrix frames
        for cm_frame in self.cm_frames:
            cm_frame.destroy()
        self.cm_frames.clear()  # Clear the list

        # Clear previous ROC curve canvas
        if self.roc_canvas_widget is not None:
            self.roc_canvas_widget.destroy()
            self.roc_canvas_widget = None

    def evaluate_model(self):
        # Clear any existing plots
        self.clear_previous_plots()

        # Step 1: Encode features and response
        self.controller.encode_features_and_response()

        # Step 2: Split the data into training and test sets
        self.controller.split_train_test_data()

        # Step 3: Train the models and make predictions
        train_success = self.controller.train_and_predict_model()

        if train_success:
            # Step 4: Evaluate the model performance and get the figures
            cm_figures, roc_figure = self.controller.evaluate_model_performance()

            # If the confusion matrix figures were successfully generated, display them
            if cm_figures:
                for model_name, cm_figure in cm_figures.items():
                    cm_frame = tk.Frame(self)
                    cm_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                    self.cm_frames.append(cm_frame)  # Keep track of the frame

                    cm_label = tk.Label(cm_frame, text=f"{model_name} Confusion Matrix")
                    cm_label.pack(side=tk.TOP)

                    cm_canvas = FigureCanvasTkAgg(cm_figure, cm_frame)
                    cm_canvas.draw()
                    cm_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # If the ROC curve figure was successfully generated, display it
            if roc_figure:
                roc_canvas = FigureCanvasTkAgg(roc_figure, self)
                roc_canvas.draw()
                roc_widget = roc_canvas.get_tk_widget()
                roc_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                self.roc_canvas_widget = roc_widget  # Keep track of the widget

    def load_xml_data(self):
        # Ask the user for the location of the XML file
        file_path = filedialog.askopenfilename(filetypes=[("XML files", "*.xml")])
        if file_path:
            # Call the provided load_dataset function with this path
            dataset = load_dataset(file_path)
            # Update the shared data with the new dataset
            self.controller.update_new_data('dataset', dataset)

    def upload_and_evaluate_data(self):
        # Fetch predictions, accuracy, and confusion matrix from the controller
        predictions_df, accuracy, cm = self.controller.read_and_predict_new_data()

        # Clear any previous accuracy labels
        if self.accuracy_label_widget is not None:
            self.accuracy_label_widget.destroy()

        # Display the accuracy as a simple number
        accuracy_text = f'Accuracy of the model: {accuracy:.2%}'  # Format as percentage
        self.accuracy_label_widget = tk.Label(self, text=accuracy_text)
        self.accuracy_label_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Clear any previous confusion matrix plots
        if self.cm_canvas_widget is not None:
            self.cm_canvas_widget.destroy()

        # Display the confusion matrix
        if cm is not None:
            cm_figure = Figure(figsize=(5, 4))
            ax_cm = cm_figure.add_subplot(111)
            sns.heatmap(cm, annot=True, cmap='viridis', fmt='d', ax=ax_cm)
            ax_cm.set_title('Confusion Matrix')
            ax_cm.set_xlabel('Predicted labels')
            ax_cm.set_ylabel('Actual labels')
            cm_canvas = FigureCanvasTkAgg(cm_figure, self)
            cm_canvas.draw()
            cm_widget = cm_canvas.get_tk_widget()
            cm_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.cm_canvas_widget = cm_widget  # Track the widget

