import tkinter as tk
from tkinter import ttk, messagebox


class Frame3(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Back button to Frame 2
        back_button = tk.Button(self, text="Back to Frame 2",
                                command=lambda: controller.show_frame("Frame2"))
        back_button.pack()

        # Machine Learning Model Selection
        self.ml_model_var = tk.StringVar(value="ML Model 1")  # Default selection
        models_frame = tk.LabelFrame(self, text="Machine Learning Model Selection")
        models_frame.pack(fill="x")
        for model in ["1 SVM", "2 Decision Tree", "3 Random Forest", "4 XGboost", "5 Neural Network"]:
            radio_button = tk.Radiobutton(models_frame, text=model, variable=self.ml_model_var, value=model)
            radio_button.pack(anchor='w')

        # Model selection confirmation button
        model_confirm_button = tk.Button(models_frame, text="Confirm Model Selection",
                                         command=self.confirm_model_selection)
        model_confirm_button.pack()

        # Train-Test Split Slider
        split_frame = tk.LabelFrame(self, text="Test Set Ratio")
        split_frame.pack(fill="x")
        self.split_slider = tk.Scale(split_frame, from_=0, to_=100, orient="horizontal")
        self.split_slider.set(80)  # Default to 80% training, 20% testing
        self.split_slider.pack(fill="x")

        # Split confirmation button
        split_confirm_button = tk.Button(split_frame, text="Confirm Split",
                                         command=self.confirm_split)
        split_confirm_button.pack()

        # ML Model Explanation Display Area
        self.model_info_text = tk.Text(self, height=4, wrap='word')
        self.model_info_text.pack(expand=True, fill="both")
        self.model_info_text.insert('end', "Select a model and click Confirm to see details here.")

        # Next Frame button
        next_frame_button = tk.Button(self, text="Next Frame",
                                      command=lambda: self.go_to_next_frame())
        next_frame_button.pack()

    def confirm_model_selection(self):
        # Assuming 'model_info' is a dict containing explanations for each model
        model_info = {
            "1 SVM": "SVM algorithm's goal is to find the optimal hyperplane that maximizes the margin between different classes.",
            "2 Decision Tree": "Decision tree algorithm recursively partition the space into subsets based on feature values, aiming to minimize impurity in each subset.",
            "3 Random Forest": "Random Forest builds multiple decision trees and aggregates their results to improve predictive performance and control over-fitting. ",
            "4 XGboost": "XGBoost belongs to a family of boosting algorithms that convert weak learners into strong ones.",
            "5 Neural Network": "Neural network employs layers of interconnected nodes, where each node combines inputs using learned weights and biases, processed through non-linear activation functions. "
        }
        selected_model = self.ml_model_var.get()
        self.controller.update_shared_data('selected_model', selected_model)
        # Update the model explanation display
        self.model_info_text.delete('1.0', 'end')
        self.model_info_text.insert('end', model_info[selected_model])

    def confirm_split(self):
        test_size = self.split_slider.get() / 100
        self.controller.update_shared_data('test_size', test_size)
        messagebox.showinfo("Split Ratio Updated", f"Training size: {100 - test_size*100}%, Test size: {test_size*100}%")

    def go_to_next_frame(self):
        test_size = self.controller.get_shared_data('test_size')
        selected_model = self.controller.get_shared_data('selected_model')
        target_column = self.controller.get_shared_data('target_column')

        # Check if test_size is a valid decimal between 0 and 1
        if test_size is not None and 0 < test_size < 1:
            # Check if selected_model is a valid string
            if isinstance(selected_model, str) and selected_model.strip():
                # Check if target_column is a valid string
                if isinstance(target_column, str) and target_column.strip():
                    self.controller.prepare_response_data()
                    # All checks passed, proceed to Frame4
                    self.controller.show_frame("Frame4")
                else:
                    messagebox.showerror("Error", "No target column has been selected.")
            else:
                messagebox.showerror("Error", "No model has been selected or the model name is invalid.")
        else:
            messagebox.showerror("Error", "The test size must be a decimal between 0 and 1.")


# The rest of your application code would go below this
# ...
