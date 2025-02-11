
import matplotlib.pyplot as plt 
import numpy as np

class NoiseAddition:
    def __init__(self, in_filepath, vmag, num_cams=24):
        self.in_filepath = in_filepath # power spectrum data from TSM 
        self.vmag = vmag # V band magnitude of star
        self.num_cams = num_cams # number of cameras on PLATO mission will either be 6, 12, 18, 24 (EOL - 22)
        self.frequencies = None
        self.clean_powers = None
        self.noisy_powers = None
    
    def add_noise(self):
        """Reads the power spectrum, adds noise, and saves a new noisy file."""
        frequencies, powers = [], []
        
        # Load data from file
        with open(self.in_filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue  # Skip comments
                parts = line.split()
                if len(parts) == 2:
                    try:
                        frequencies.append(float(parts[0]))
                        powers.append(float(parts[1]))
                    except ValueError:
                        continue

        # Convert to NumPy arrays
        self.frequencies = np.array(frequencies)
        self.clean_powers = np.array(powers)  

        # Noise calculations
        time = 25  # Integration time (fixed)
        cam_scale = 408.51 * self.num_cams ** (-0.98)  
        bgshot = cam_scale * 10**(-0.4 * (11.0 - self.vmag))  
        rms = np.sqrt(bgshot * 1.0e6 / (2.0 * 50.0))  
        rms_at_t = rms * np.sqrt(50.0 / time)  

        # Generate noisy power values
        powers_shot_noise = rms_at_t + self.clean_powers  
        random_array = np.random.uniform(0, 1, size=self.clean_powers.shape)  
        self.noisy_powers = -powers_shot_noise * np.log(random_array)  

        # Save noisy data
        out_file_path = f"{self.in_filepath.replace('.pow', '')}_cams{self.num_cams:.0f}_vmag{self.vmag:.1f}.pow"
        with open(out_file_path, "w") as file:
            file.write("# Frequency [muHz], Power [ppm^2/muHz]\n")
            for freq, power in zip(self.frequencies, self.noisy_powers):
                file.write(f"{freq:.16e} {power:.16e}\n")
        
        print(f"Noisy data saved to {out_file_path}")

    def plot_noise(self):
        """Plots the original and noisy power spectra."""
        if self.frequencies is None or self.noisy_powers is None:
            print("Error: No data available. Run `add_noise()` first.")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.frequencies, self.clean_powers, label="Clean Power", alpha=0.7)
        plt.plot(self.frequencies, self.noisy_powers, label="Noisy Power", alpha=0.7, linestyle="dashed")
        plt.title(f"Power Spectrum (cams = {self.num_cams}) (Vmag = {self.vmag})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Frequency [µHz]")
        plt.ylabel("Power [ppm²/µHz]")
        plt.legend()
        plt.grid(True)
        plt.show()


