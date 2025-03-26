import os
import re
import numpy as np
import matplotlib.pyplot as plt

class StarDataHandler:
    def __init__(self):
        pass

    def extract_star_index(self, filepath):
        """Extracts the star index from the input filepath."""
        match = re.search(r'(?:spectrum_)(\d+)', filepath)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Invalid filename format: {filepath}. Expected 'spectrum_X'.")

    def get_vmag_from_metadata(self, filepath):
        """Extracts the V-band magnitude (V:) from the metadata in the .pow file."""
        try:
            with open(filepath, 'r', encoding='latin-1') as f:  # Handle special characters (µHz)
                for line in f:
                    if line.startswith('# V:'):
                        try:
                            vmag = float(line.split(':')[1].strip())
                            return vmag
                        except (ValueError, IndexError):
                            raise ValueError(f"Invalid V-band magnitude format in {filepath}.")
            raise ValueError(f"V-band magnitude not found in metadata of {filepath}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
    
    def get_logL_from_metadata(self, filepath):
        """Extracts the logL (log of luminosity) from the metadata."""
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                for line in f:
                    if line.startswith('# logL:'):
                        try:
                            logL = float(line.split(':')[1].strip())
                            return logL
                        except (ValueError, IndexError):
                            raise ValueError(f"Invalid logL format in {filepath}.")
            raise ValueError(f"logL not found in metadata of {filepath}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")

    def calculate_distance(self, vmag, logL):
        """Calculate the distance to the star using the distance modulus."""

        # Calculate the absolute magnitude M from logL
        M = -2.5 * logL + 4.75  # Absolute magnitude
        
        # Distance modulus: m - M = 5 log10(d) - 5
        distance = 10 * 10 ** ((vmag - M) / 5)
        return distance


class NoiseAddition:
    def __init__(self, in_filepath, num_cams=24, vmag_option="metadata", manual_vmag=None, num_vmags=3, vmag_lower=7, vmag_upper=10.5):
        """
        Parameters:
        - in_filepath: Path to the input power spectrum file.
        - num_cams: Number of cameras on the PLATO mission.
        - vmag_option: 
            "metadata" -> Extract Vmag from metadata.
            "random" -> Generate random Vmag values between a lower and upper limit.
            "manual" -> Use manually specified Vmag (requires manual_vmag argument).
            "steps" -> Generate Vmag values evenly distributed between vmag_lower and vmag_upper.
        - manual_vmag: If vmag_option="manual", this value is used as the Vmag.
        - num_vmags: Number of Vmag values to generate per spectrum (applies to "random" and "steps").
        - vmag_lower: Lower limit of the Vmag range (applies to "random" and "steps").
        - vmag_upper: Upper limit of the Vmag range (applies to "random" and "steps").
        """
        self.in_filepath = in_filepath  
        self.num_cams = num_cams  
        self.frequencies = None
        self.clean_powers = None
        self.noisy_powers = []
        self.vmag_option = vmag_option
        self.manual_vmag = manual_vmag
        self.num_vmags = num_vmags
        self.vmag_lower = vmag_lower
        self.vmag_upper = vmag_upper

        # Retrieve luminosity from metadata (it remains constant)
        data_handler = StarDataHandler()
        self.logL = data_handler.get_logL_from_metadata(in_filepath)  # Get logL for luminosity calculation

        if self.vmag_option == "metadata":
            self.vmag = data_handler.get_vmag_from_metadata(in_filepath)
        elif self.vmag_option == "manual":
            if manual_vmag is None:
                raise ValueError("Manual Vmag option selected, but no manual Vmag provided.")
            self.vmag = manual_vmag
        elif self.vmag_option == "random":
            self.vmag = None  # Will generate random values in add_noise()
        elif self.vmag_option == "steps":
            self.vmag = None  # Will generate stepped values in add_noise()
        else:
            raise ValueError("Invalid vmag_option. Choose 'metadata', 'random', 'manual', or 'steps'.")

    def add_noise(self):
        frequencies, powers = [], []
        metadata = []
        output_dir = "NoisyData"
        os.makedirs(output_dir, exist_ok=True)
        in_file = os.path.basename(self.in_filepath)

        with open(self.in_filepath, 'r', encoding='latin-1') as f:
            for _ in range(39):
                metadata.append(next(f))
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        frequencies.append(float(parts[0]))
                        powers.append(float(parts[1]))
                    except ValueError:
                        continue

        self.frequencies = np.array(frequencies)
        self.clean_powers = np.array(powers)

        if self.vmag_option == "metadata" or self.vmag_option == "manual":
            vmag_values = [self.vmag]
        elif self.vmag_option == "random":
            vmag_values = np.random.uniform(self.vmag_lower, self.vmag_upper, size=self.num_vmags)
        elif self.vmag_option == "steps":
            vmag_values = np.linspace(self.vmag_lower, self.vmag_upper, self.num_vmags)

        for vmag in vmag_values:
            cam_scale = 408.51 * self.num_cams ** (-0.98)
            bgshot = cam_scale * 10**(-0.4 * (11.0 - vmag))
            powers_shot_noise = bgshot + self.clean_powers
            random_array = np.random.uniform(0, 1, size=self.clean_powers.shape)
            noisy_powers = -powers_shot_noise * np.log(random_array)

            self.noisy_powers.append((vmag, noisy_powers))

            # Calculate the distance for the star based on the current Vmag
            distance = StarDataHandler().calculate_distance(vmag, self.logL)
            print(f"Distance to star (Vmag = {vmag}): {distance:.2f} parsecs")

            out_file_name = f"{in_file.replace('.pow', '')}_cams{self.num_cams:.0f}_vmag{vmag:.2f}.pow"
            out_file_path = os.path.join(output_dir, out_file_name)

            with open(out_file_path, "w") as file:
                file.writelines(metadata)
                for freq, power in zip(self.frequencies, noisy_powers):
                    file.write(f"{freq:.16e} {power:.16e}\n")

            print(f"Noisy data saved to {out_file_path}")

    def plot_noise(self):
        """Plots the original and noisy power spectra."""
        if self.frequencies is None or not self.noisy_powers:
            print("Error: No data available. Run `add_noise()` first.")
            return

        plt.figure(figsize=(8, 6))
        plt.plot(self.frequencies, self.clean_powers, label="Clean Power", alpha=0.7)

        # Plot all generated noisy spectra
        for vmag, noisy_powers in self.noisy_powers:
            # Calculate the distance for the current Vmag
            distance = StarDataHandler().calculate_distance(vmag, self.logL)
            plt.plot(self.frequencies, noisy_powers, label=f"Noisy Power (Vmag={vmag:.1f}, Distance={distance:.2f} parsecs)", alpha=0.7, linestyle="dashed")

        plt.title(f"Noisy Power Spectrum (cams = {self.num_cams})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Frequency [µHz]")
        plt.ylabel("Power [ppm²/µHz]")
        plt.legend()
        plt.grid(True)
        plt.show()
