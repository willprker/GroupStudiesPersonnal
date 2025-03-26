import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class BestFilters:
    def __init__(self, filepath, output_dir="bestFilters"):
        self.filepath = filepath
        self.filename = os.path.splitext(os.path.basename(filepath))[0]
        self.frequencies = None
        self.data = None
        self.best_df = None
        self.output_dir = output_dir

    def load_file(self):
        try:
            df = pd.read_csv(self.filepath)
            self.frequencies = df.iloc[:, 0].values.astype(float)
            self.data = df.iloc[:, 1:].values.astype(float)
            self.columns = df.columns[1:]
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            return
        except Exception as e:
            print(f"An error occurred during file loading: {e}")
            return

    def process_and_plot(self):
        """Loads, processes, and plots the best three filters."""
        if not hasattr(self, 'filename') or not self.filename:
            self.filename = os.path.splitext(os.path.basename(self.filepath))[0]

        self.load_file()
        if self.data is None:
            return 

        self.best_three_filters()
        if self.best_df is None:
            return

        # self.plot_best_three()

    def best_three_filters(self):
        """Finds the best three filters, saves them to a new CSV, and includes frequencies."""
        if self.data is None:
            return None

        dataframe = pd.DataFrame(self.data, columns=self.columns)

        max_values = dataframe.max()
        
        best_columns = max_values.nlargest(3).index.tolist()

        sorted_columns = sorted(best_columns, key=lambda col: max_values[col], reverse=True)

        self.best_df = dataframe[sorted_columns].copy()
        self.best_df.insert(0, "Frequency (µHz)", self.frequencies)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_filepath = os.path.join(self.output_dir, f"{self.filename}.csv")
        self.best_df.to_csv(output_filepath, index=False)
        print(f"Best three filters saved to: {output_filepath}")

    def plot_best_three(self):
        """Plots the best three filters in descending order with maximum value in the legend."""
        if self.best_df is None:
            return

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        for i, column_name in enumerate(self.best_df.columns[1:]):
            y_values = self.best_df[column_name]
            max_value = y_values.max()
            axes[i].plot(self.best_df["Frequency (µHz)"], y_values, label=f"{column_name} (Max: {max_value:.2f})")
            axes[i].legend(loc="upper right")
            axes[i].set_ylabel("Signal")
            axes[i].set_title(f"Best Filter {i+1}")

        axes[-1].set_xlabel("Frequency (µHz)")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_filepath = os.path.join(self.output_dir, f"{self.filename}_best_plot.png")
        plt.savefig(output_filepath)
        print(f"Plot saved to: {output_filepath}")
        # plt.show()
