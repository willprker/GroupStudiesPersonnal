import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import os
import pandas as pd

class PowerSpectrumProcessor:
    def __init__(self, filepath, output_dir=".", output_csv="combined_filtered_output.csv"):
        self.filepath = filepath
        self.frequencies = None
        self.signal = None
        self.results = {}
        self.output_dir = output_dir  # Store output directory
        self.output_csv = output_csv  # Store CSV filename
        self.data_directory = None # Store the directory where the filtered files are

    
    def lorentzian(self, x, x0, a, gam):
        return a * gam**2 / (gam**2 + (x - x0)**2)
    
    def load_power_spectrum(self):
        """Loads frequency and power spectrum data from a .pow file."""
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
            print(f"Error: File not found at {self.filepath}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def granulation_lensing(self, A, B, C):
        """
        Establishes the granulation spectra on the power spectrum.

        Parameters:
            A: Granulation signal amplitude
            B: Characteristic timescale (often the turnover frequency of granulation in Hz)
            C: Generic slope parameter
        """
        f = self.frequencies  # Use the loaded frequencies from the data
        gran_spec = A / (1 + (B * f)**C)
        return gran_spec

    def subtract_granulation_lensing(self, A, B, C):
        """
        Subtracts the granulation lensing effect from the loaded power spectrum data.

        Parameters:
            A: Granulation signal amplitude
            B: Characteristic timescale
            C: Slope parameter
        """
        if self.signal is None:
            print("Error: Signal data is not loaded.")
            return

        gran_spectrum = self.granulation_lensing(A, B, C)
        self.signal -= gran_spectrum  # Subtract from the power spectrum
    
    def process_filter(self, spacing, queue):
        """Performs convolution for a specific filter and saves the result."""
        if self.signal is None:
            print("Error: Signal data is not loaded.")
            return

        x = np.linspace(-80000, 80000, 160000)
        gamma = 200  # HWHM
        a = 1
        
        x0_values = [-spacing, spacing, -spacing*2 - spacing, spacing*2 + spacing, 
                     -spacing*3 - spacing*2, spacing*3 + spacing*2, -spacing*4 - spacing*3, spacing*4 + spacing*3]
        
        combined_filter = sum(self.lorentzian(x, x0, a, gamma) for x0 in x0_values)
        output = np.convolve(self.signal[15000:700000], combined_filter, mode='valid')
        queue.put((spacing, output))
    
    def run_filtering(self, subtract_granulation=False, A=None, B=None, C=None):
        """
        Runs filtering operations, optionally subtracting granulation lensing first.

        Parameters:
            subtract_granulation: If True, subtract granulation lensing before filtering.
            A, B, C: Parameters for granulation lensing if subtraction is enabled.
        """
        if self.frequencies is None or self.signal is None:
            print("Error: Data not loaded. Run load_power_spectrum() first.")
            return

        if subtract_granulation:
            if A is None or B is None or C is None:
                raise ValueError("A, B, and C must be provided for granulation subtraction.")
            self.subtract_granulation_lensing(A, B, C)
        
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
    
    def save_results(self):  # Restored to original name
        """Saves the filtered outputs to individual text files."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for spacing, output in self.results.items():
            filename = f"{self.output_dir}/filtered_output_{(spacing*2)/100}.txt"
            np.savetxt(filename, output)
        print("Filtered outputs saved successfully.")

    def combine_and_save_to_csv(self):
        """Combines all filtered outputs into a single CSV file."""

        if self.data_directory is None:
            raise ValueError("data_directory is not set. Call load_filtered_data_from_directory() first.")

        # Get all filtered output files and sort
        files = sorted(
            [
                f
                for f in os.listdir(self.data_directory)
                if f.startswith("filtered_output_") and f.endswith(".txt")
            ],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )

        # Initialize a dictionary to store data
        data_dict = {}

        # Load each file and store data
        for file in files:
            filepath = os.path.join(self.data_directory, file)  # Use data_directory here
            data = np.loadtxt(filepath)
            data_dict[file] = data

        # Convert dictionary to DataFrame
        df = pd.DataFrame(data_dict)

        output_path = os.path.join(self.data_directory, self.output_csv)
        df.to_csv(output_path, index=False)
        print(f"Combined CSV saved as: {output_path}")
        self.df = df # adds the dataframe to the class so that it can be used for plotting

    def plot_filtered_outputs(self):
        """Plots all filtered outputs."""

        if not hasattr(self, 'df'): # checks if the dataframe exists within the class
            raise ValueError("Dataframe does not exist. Call combine_and_save_to_csv() first.")

        num_plots = len(self.df.columns) # changed to look at the dataframe
        fig, axes = plt.subplots(nrows=num_plots, figsize=(8, num_plots * 2), sharex=True)

        if num_plots == 1:
            axes = [axes]

        for ax, file in zip(axes, self.df.columns): # changed to look at the dataframe
            ax.plot(self.df[file], label=file) # changed to look at the dataframe
            ax.set_ylabel("Amplitude")
            ax.legend()

        axes[-1].set_xlabel("Sample Index")
        plt.suptitle("Filtered Outputs for Different Spacings (Sorted by Filter Number)")
        plt.tight_layout()
        plt.show()

    def load_filtered_data_from_directory(self, data_directory):
        """Loads the filtered data from a directory."""

        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Directory not found: {data_directory}")

        self.data_directory = data_directory
