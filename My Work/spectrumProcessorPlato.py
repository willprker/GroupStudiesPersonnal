import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import peak_prominences

class PowerSpectrumProcessor:
    def __init__(self, filepath, output_dir="filterOutput"):
        ''' Initialize the PowerSpectrumProcessor with the filepath of the .pow file and the output directory for saving results '''
        self.filepath = filepath
        self.frequencies = None
        self.signal = None
        self.results = {}
        self.output_dir = output_dir

    def lorentzian(self, x, x0, a, gam):
        ''' Lorentzian function for filtering '''
        return a * gam**2 / (gam**2 + (x - x0)**2)

    def load_power_spectrum(self):
        ''' Load the power spectrum data from the .pow file '''
        try:
            metadata = []
            frequencies = []
            powers = []
            with open(self.filepath, 'r', encoding='latin-1') as f:
                for _ in range(39):
                    metadata.append(next(f))

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
            raise FileNotFoundError(f"File not found at {self.filepath}")
        except Exception as e:
            raise Exception(f"An error occurred: {e}")


    def process_filter(self, spacing):
        ''' Process the filtering operation for a given spacing and return the result '''
        if self.signal is None:
            raise ValueError("Signal data is not loaded.")

        x = np.linspace(-44100, 44100, 88200)
        gamma = 127
        a = 10

        x0_values = [-spacing, spacing, -spacing*2 - spacing, spacing*2 + spacing, 
                    -spacing*3 - spacing*2, spacing*3 + spacing*2, -spacing*4 - spacing*3, spacing*4 + spacing*3]

        combined_filter = sum(self.lorentzian(x, x0, a, gamma) for x0 in x0_values)

        start_signal = int(1500 * 63.072)  # 94608
        end_signal = int(7000 * 63.072)    # 441504

        output = np.convolve(self.signal[(start_signal - 44100):(end_signal + 44100)], combined_filter, mode='valid')

        start_index = start_signal
        end_index = start_index + len(output)
        
        output_frequencies = self.frequencies[start_index:end_index]

        return spacing, output, output_frequencies

    def run_filtering(self):
        ''' Run the filtering operation for different spacings sequentially '''
        if self.frequencies is None or self.signal is None:
            raise ValueError("Data not loaded. Run load_power_spectrum() first.")

        spacing_points = np.arange(64, 84, 1)
        spacing_microhz = spacing_points * 63.072

        for spacing in spacing_microhz:
            spacing, output, output_frequencies = self.process_filter(spacing)
            self.results[spacing] = (output, output_frequencies)

        print("All filtering operations completed.")

    def subtract_baseline(self, data):
        ''' Subtract the baseline from the data using the bottom line method '''
        inverted_data = -data
        peaks, _ = find_peaks(inverted_data, distance=12700)
        prominences, _, _ = peak_prominences(inverted_data, peaks)
        prominent_peaks_mask = prominences > 10
        prominent_peaks = peaks[prominent_peaks_mask] 

        if 0 not in prominent_peaks:
            prominent_peaks = np.insert(prominent_peaks, 0, 0)

        bottom_line_indices = prominent_peaks
        bottom_line_values = data[bottom_line_indices] 

        f = interp1d(bottom_line_indices, bottom_line_values, kind='linear', fill_value="extrapolate")
        interpolated_bottom_line = f(np.arange(len(data)))
        corrected_data = data - interpolated_bottom_line
 
        return corrected_data

    def run_baseline_subtraction(self):
        ''' Run the baseline subtraction on the filtered outputs '''
        if not self.results:
            raise ValueError("No filtered data available. Run run_filtering() first.")

        for spacing, (output, output_frequencies) in self.results.items():
            self.results[spacing] = (self.subtract_baseline(output), output_frequencies)

        print("Baseline subtraction completed for all filtered outputs.")

    def save_results(self, subtract_baseline=False):
        ''' Save the filtered results to a CSV file containing the frequency and amplitude data per filter '''
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        all_data = []

        for spacing, (output, output_frequencies) in self.results.items():
            frequency_label = f"{(spacing * 2) / 63.072} microHz"
            temp_df = pd.DataFrame({'Frequency (µHz)': output_frequencies, frequency_label: output})
            all_data.append(temp_df)

        final_df = all_data[0]
        for i in range(1, len(all_data)):
            final_df = pd.merge(final_df, all_data[i], on='Frequency (µHz)', how='outer')

        filename = os.path.splitext(os.path.basename(self.filepath))[0]
        output_filename = f"{filename}.csv"
        output_path = os.path.join(self.output_dir, output_filename)
        final_df.to_csv(output_path, index=False)
        print(f"Combined CSV saved at: {output_path}")

    def plot_filtered_outputs(self, subtract_baseline=False):
        ''' Plot the filtered outputs for different spacings '''
        if not self.results:
            raise ValueError("No filtered data available. Run run_filtering() first.")

        num_plots = len(self.results)
        fig, axes = plt.subplots(nrows=num_plots, figsize=(8, num_plots * 2), sharex=True)

        if num_plots == 1:
            axes = [axes]

        for ax, (spacing, (output, output_frequencies)) in zip(axes, self.results.items()):
            frequency = (spacing*2)/63.072
            ax.plot(output_frequencies, output, label=f"{frequency} microHz")
            ax.set_ylabel("Amplitude")
            ax.set_xlabel("Frequency (µHz)")
            ax.legend()

        plt.suptitle("Filtered Outputs for Different Spacings")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()