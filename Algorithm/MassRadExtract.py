import pandas as pd
import os
import re

class M_R_Est:
    def __init__(self, csv_path, folder_path, processor_class):
        """        
        :param csv_path: Path to the CSV file.
        :param folder_path: Path to the folder containing metadata files.
        :param processor_class: The main processing class that takes v_max, delta_v, BV, and FeH.
        """
        self.csv_path = csv_path
        self.folder_path = folder_path
        self.processor_class = processor_class
        self.metadata_dict = {}

    class MetadataExtractor:
        """Extracts values from metadata files."""
        
        @staticmethod
        def get_vmag_from_metadata(filepath):
            """Extracts the V-band magnitude (V:) from metadata."""
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    for line in f:
                        if line.startswith('# V:'):
                            try:
                                return float(line.split(':')[1].strip())
                            except (ValueError, IndexError):
                                raise ValueError(f"Invalid V-band magnitude format in {filepath}.")
                raise ValueError(f"V-band magnitude not found in {filepath}.")
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {filepath}")
        
        @staticmethod
        def get_bmag_from_metadata(filepath):
            """Extracts the B-band magnitude (B:) from metadata."""
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    for line in f:
                        if line.startswith('# B:'):
                            try:
                                return float(line.split(':')[1].strip())
                            except (ValueError, IndexError):
                                raise ValueError(f"Invalid B-band magnitude format in {filepath}.")
                raise ValueError(f"B-band magnitude not found in {filepath}.")
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {filepath}")
        
        @staticmethod
        def get_feh_from_metadata(filepath):
            """Extracts the metallicity [M/H] from metadata."""
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    for line in f:
                        if line.startswith('# [M/H]:'):
                            try:
                                return float(line.split(':')[1].strip())
                            except (ValueError, IndexError):
                                raise ValueError(f"Invalid metallicity format in {filepath}.")
                raise ValueError(f"Metallicity not found in {filepath}.")
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {filepath}")

    def extract_metadata(self):
        """Extracts metadata from .pow files and stores it in a dictionary."""
        metadata_extractor = self.MetadataExtractor()
        
        # Find unique spectra (spectrum_X)
        pow_files = [f for f in os.listdir(self.folder_path) if f.endswith(".pow")]
        unique_spectra = set(re.match(r"(spectrum_\d+)", f).group(1) for f in pow_files if re.match(r"(spectrum_\d+)", f))
        
        for spectrum_base in unique_spectra:
            matching_files = [f for f in pow_files if f.startswith(spectrum_base)]
            if matching_files:
                file_path = os.path.join(self.folder_path, matching_files[0])  # Pick the first match
                if os.path.exists(file_path):
                    try:
                        BV = metadata_extractor.get_bmag_from_metadata(file_path) - metadata_extractor.get_vmag_from_metadata(file_path)
                        FeH = metadata_extractor.get_feh_from_metadata(file_path)
                        self.metadata_dict[spectrum_base] = {"BV": BV, "FeH": FeH}
                    except (ValueError, FileNotFoundError) as e:
                        print(f"Error processing {spectrum_base}: {e}")

    def update_csv(self):
        """Updates the CSV file with extracted metadata and processed outputs."""
        # Read and standardize CSV
        df = pd.read_csv(self.csv_path)
        # df.columns = df.columns.str.strip().str.lower()  # Normalize column names
        
        required_columns = {"filename", "nu_max", "delta nu"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise KeyError(f"Missing columns in CSV: {missing_columns}\nAvailable: {df.columns.tolist()}")
        
        # Extract spectrum base (spectrum_X) from spectrum_id
        df["spectrum_base"] = df["filename"].apply(lambda x: re.match(r"(spectrum_\d+)", x).group(1) if re.match(r"(spectrum_\d+)", x) else x)

        # Ensure the columns exist before assigning values
        for col in ["mass guess", "mass error", "radius guess", "radius error"]:
            if col not in df.columns:
                df[col] = None  # Initialize with NaN values

        # Process each row
        for index, row in df.iterrows():
            spectrum_base = row["spectrum_base"]
            if spectrum_base in self.metadata_dict:
                V_max = row["nu_max"]
                Delta_V = row["delta nu"]
                BV = self.metadata_dict[spectrum_base]["BV"]
                FeH = self.metadata_dict[spectrum_base]["FeH"]

                # Compute mass and radius
                processor = self.processor_class(V_max, Delta_V, BV, FeH)
                mass, mass_err, radius, radius_err = processor.calc_mass_and_rad()

                # Assign values
                df.at[index, "mass guess"] = mass
                df.at[index, "mass error"] = mass_err
                df.at[index, "radius guess"] = radius
                df.at[index, "radius error"] = radius_err

        # Save the updated CSV
        df.to_csv(self.csv_path, index=False)
        print(f"CSV file '{self.csv_path}' updated successfully.")


        df.drop(columns=["Mass Estimate Error", "Mass Estimate", "Radius Estimate", "Radius Estimate Error"], inplace=True, errors='ignore')

        df.to_csv(self.csv_path, index=False)
        print(f"CSV file '{self.csv_path}' updated successfully.")

    def run(self):
        """Runs the full pipeline: extract metadata and update CSV."""
        print("Extracting metadata...")
        self.extract_metadata()
        print("Updating CSV with processed outputs...")
        self.update_csv()
        print("Process complete.")
