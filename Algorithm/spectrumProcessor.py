import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import os
import pandas as pd
from scipy.signal import find_peaks

class PowerSpectrumProcessor:
    def __init__(self, filepath, output_dir=".", output_csv="combinedFilteredData.csv"):
        self.filepath = filepath
        self.frequencies = None
        self.signal = None
        self.results = {}
        self.output_dir = output_dir
        self.output_csv = output_csv

    def lorentzian(self, x, x0, a, gam):
        return a * gam**2 / (gam**2 + (x - x0)**2)

    def load_power_spectrum(self):
        try:
            frequencies = []
            powers = []
            with open(self.filepath, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            frequencies.append(float(parts[0]))
                            powers.append(float(parts[1]))
                        except ValueError:
                            print(f"Skipping invalid line: {line.strip()}")

            self.frequencies = np.array(frequencies)
            self.signal = np.array(powers)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.filepath}")  # Raise exception
        except Exception as e:
            raise Exception(f"An error occurred: {e}")  # Raise exception

    def process_filter(self, spacing, queue):
        if self.signal is None:
            raise ValueError("Signal data is not loaded.")  # Raise exception

        x = np.linspace(-80000, 80000, 160000)
        gamma = 200
        a = 1

        x0_values = [-spacing, spacing, -spacing*2 - spacing, spacing*2 + spacing, 
                    -spacing*3 - spacing*2, spacing*3 + spacing*2, -spacing*4 - spacing*3, spacing*4 + spacing*3]

        combined_filter = sum(self.lorentzian(x, x0, a, gamma) for x0 in x0_values)
        output = np.convolve(self.signal[15000:700000], combined_filter, mode='valid')
        queue.put((spacing, output))

    def run_filtering(self):
        if self.frequencies is None or self.signal is None:
            raise ValueError("Data not loaded. Run load_power_spectrum() first.")  # Raise exception

        spacing_points = np.arange(30, 103, 3)
        spacing_microhz = spacing_points * 100

        processes = []
        queue = Queue()

        for spacing in spacing_microhz:
            process = Process(target=self.process_filter, args=(spacing, queue))
            process.start()
            processes.append(process)

        for _ in range(len(processes)):
            spacing, output = queue.get()
            self.results[spacing] = output

        for process in processes:
            process.join()

        print("All filtering operations completed.")

    def subtract_baseline(self, data):
        """Subtracts the baseline from a given data array."""
        peaks, _ = find_peaks(-data, distance=10000)  # Adjust distance as needed

        df = pd.DataFrame(data, columns=['values'])
        df['bottom_line'] = np.nan
        df.loc[df.index.isin(peaks), 'bottom_line'] = df['values']
        df['smooth_bottom_line'] = df['bottom_line'].interpolate(method='linear')

        subtracted_data = data - df['smooth_bottom_line']
        return subtracted_data

    def run_baseline_subtraction(self):
        """Applies baseline subtraction to all filtered outputs."""
        if not self.results:
            raise ValueError("No filtered data available. Run run_filtering() first.")

        for spacing, output in self.results.items():
            self.results[spacing] = self.subtract_baseline(output)

        print("Baseline subtraction completed for all filtered outputs.")

    def save_results(self, subtract_baseline=False):
        """Combines filtered outputs (optionally with baseline subtraction) into a CSV."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        modified_results = {}
        for spacing, output in self.results.items():
            frequency = (spacing * 2) / 100
            modified_results[f"{frequency} microHz"] = output

        df = pd.DataFrame(modified_results)
        output_path = os.path.join(self.output_dir, self.output_csv)
        df.to_csv(output_path, index=False)
        print(f"Combined CSV saved at: {output_path}")


    def plot_filtered_outputs(self, subtract_baseline=False):
        """Plots all filtered outputs (optionally with baseline subtraction)."""

        if not self.results:
            raise ValueError("No filtered data available. Run run_filtering() first.")

        num_plots = len(self.results)
        fig, axes = plt.subplots(nrows=num_plots, figsize=(8, num_plots * 2), sharex=True)

        if num_plots == 1:
            axes = [axes]

        for ax, (spacing, output) in zip(axes, self.results.items()):
            frequency = (spacing * 2) / 100
            ax.plot(output, label=f"{frequency} microHz")
            ax.set_ylabel("Amplitude")
            ax.set_xlabel("Sample Index")
            ax.legend()

        plt.suptitle("Filtered Outputs for Different Spacings")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()