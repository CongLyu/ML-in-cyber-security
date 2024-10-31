import tkinter as tk
from tkinter import ttk
import pandas as pd


class Frame2(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.temp_selected_var = None
        self.controller = controller
        self.pack(fill="both", expand=True)

        # Button to go back to Frame1
        back_button = tk.Button(self, text="Back to Frame 1",
                                command=lambda: controller.show_frame("Frame1"))
        back_button.pack(side="top")

        # Entry widget to display dataset name (this is just for display,
        # not editing)
        self.dataset_name_entry = tk.Entry(self)
        self.dataset_name_entry.pack(side="top", fill="x")

        # Table to display the head of the dataset
        self.table = ttk.Treeview(self)
        self.table.pack(side="top", fill="both", expand=True)

        # Scrollbar for the table
        self.scrollbar = tk.Scrollbar(self, orient="vertical",
                                      command=self.table.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=self.scrollbar.set)

        # Listbox to display all column names
        self.column_listbox = tk.Listbox(self, selectmode="multiple")
        self.column_listbox.pack(side="left", fill="y", expand=True)

        # Drop Column Button
        drop_column_button = tk.Button(self, text="Drop Column",
                                       command=self.drop_columns)
        drop_column_button.pack(side="left")

        # Variable to store the name of the selected response variable
        self.response_var = tk.StringVar(value=None)

        # Label for the response variable selection
        label = tk.Label(self, text="Select the response variable:")
        label.pack()

        # Confirm Selection button
        self.confirm_button = tk.Button(self, text="Confirm Selection",
                                        command=self.confirm_response_variable)
        self.confirm_button.pack()

        # Create a radiobutton for each column in the dataset
        self.create_response_variable_selector()

        # Button to go to the next frame (Frame3)
        next_frame_button = tk.Button(self, text="Next Frame",
                                      command=lambda: controller.show_frame(
                                          "Frame3"))
        next_frame_button.pack(side="bottom")

    def update_display(self):
        # This method updates the Frame2 display with data
        # Call the methods that update the widgets in Frame2
        self.update_column_listbox()  # Update your column listbox
        self.create_response_variable_selector()  # Update your radiobuttons for response variable
        self.update_table()

    def update_table(self):
        # Clear previous table content
        for i in self.table.get_children():
            self.table.delete(i)
        # Attempt to get 'dataset_reduced' from shared data
        dataset_reduced = self.controller.get_shared_data('dataset_reduced')
        # If 'dataset_reduced' is None, fall back to 'dataset'
        dataset = dataset_reduced if dataset_reduced is not None else self.controller.get_shared_data(
            'dataset')
        if dataset is not None:
            # dataframe can be retrieved by controller
            columns = list(dataset.columns)

            # Attempt to get 'target_column' from shared data
            target_column = self.controller.get_shared_data('target_column')

            self.table['columns'] = columns
            if dataset is not None:
                for col in columns:
                    if col == target_column:
                        # Apply the target column style to the heading of the target column
                        self.table.heading(col, text=f"* {col} *")
                    else:
                        self.table.heading(col, text=col)

                for row in dataset.head().itertuples():
                    self.table.insert('', 'end', values=row[1:])

    def highlight_column(self, col):
        # This function could be used to highlight the entire column in the table when clicked
        # Implementation will depend on the specifics of the table widget being used
        pass

    def update_column_listbox(self):
        # Attempt to get 'dataset_reduced' and 'target_column' from shared data
        dataset_reduced = self.controller.get_shared_data('dataset_reduced')
        target_column = self.controller.get_shared_data('target_column')
        # If 'dataset_reduced' is None, fall back to 'dataset'
        dataset = dataset_reduced if dataset_reduced is not None else self.controller.get_shared_data(
            'dataset')

        self.column_listbox.delete(0, 'end')
        if dataset is not None:
            for col in dataset.columns:
                if col != target_column:  # Exclude the response variable from the list
                    self.column_listbox.insert('end', col)

    def drop_columns(self):
        selected_indices = self.column_listbox.curselection()
        selected_columns = [self.column_listbox.get(i) for i in
                            selected_indices]
        self.controller.drop_dataset_column(selected_columns)
        self.update_table()
        self.update_column_listbox()
        # Update the shared data with the new dataset
        self.controller.update_shared_data('dataset_reduced',
                                           self.controller.get_shared_data(
                                               'dataset_reduced'))
        self.create_response_variable_selector()
        self.update_column_listbox()

    def create_response_variable_selector(self):
        # Attempt to get 'dataset_reduced' from shared data
        dataset_reduced = self.controller.get_shared_data('dataset_reduced')
        # If 'dataset_reduced' is None, fall back to 'dataset'
        dataset = dataset_reduced if dataset_reduced is not None else self.controller.get_shared_data(
            'dataset')

        # Clear previous radio buttons
        for widget in self.winfo_children():
            if isinstance(widget, tk.Radiobutton):
                widget.destroy()

        # Proceed only if a dataset is available
        if dataset is not None:
            for col in dataset.columns:
                radiobutton = tk.Radiobutton(self, text=col,
                                             variable=self.response_var,
                                             value=col,
                                             command=self.update_response_variable)
                radiobutton.pack(anchor='w')

    def update_response_variable(self):
        # Temporarily get the selected value from the radio buttons
        self.temp_selected_var = self.response_var.get()
        # No immediate update to shared data here; wait for confirmation

    def confirm_response_variable(self):
        # Use the temporarily selected variable as the final choice
        selected_var = self.temp_selected_var
        self.controller.update_shared_data('target_column', selected_var)

        # Attempt to get 'dataset_reduced' from shared data
        dataset_reduced = self.controller.get_shared_data('dataset_reduced')
        # If 'dataset_reduced' is None, fall back to 'dataset'
        dataset = dataset_reduced if dataset_reduced is not None else self.controller.get_shared_data(
            'dataset')

        if dataset is not None:
            # Set the new response dataset and update covariates accordingly
            response = dataset[[selected_var]]
            covariates = dataset.drop(columns=[selected_var])

            # Update the shared data with the newly confirmed response and covariates
            self.controller.update_shared_data('dataset_response', response)
            self.controller.update_shared_data('dataset_covariates', covariates)
            self.update_display()
