import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re


class MassComparison:
    """
    A class for comparing guessed stellar masses with actual masses from metadata files,
    processing CSV files, and generating visualizations of the results.
    """

    def __init__(self, results_csv=None, data_directory=None, filename_column="filename"):
        """
        Initialize the MassComparison object.

        Args:
            results_csv (str): Path to the CSV file containing guessed masses
            data_directory (str): Directory containing the actual data files with metadata
            filename_column (str): Name of the column in CSV that contains filenames
        """
        self.results_csv = results_csv
        self.data_directory = data_directory
        self.filename_column = filename_column
        self.processed_df = None
        self.comparison_results = None
        self.comparison_df = None

    def process_csv(self, input_file=None, output_file=None):
        """
        Process a CSV file without modifying any mass values.
        Simply reads the input CSV and creates a copy as output.
        
        Args:
            input_file (str): Path to the input CSV file (defaults to self.results_csv if None)
            output_file (str): Path to the output CSV file (default: adds '_processed' to input filename)
            
        Returns:
            pandas.DataFrame: The processed DataFrame
        """
        # Use instance variable if input_file not provided
        if input_file is None:
            input_file = self.results_csv
            if input_file is None:
                raise ValueError("No input file specified")
        
        # Generate default output filename if not provided
        if output_file is None:
            file_name, file_ext = os.path.splitext(input_file)
            output_file = f"{file_name}_processed{file_ext}"
        
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check if required columns exist
        if 'mass guess' not in df.columns:
            raise ValueError("CSV must contain 'mass guess' column")
        
        # Save the dataframe to a new CSV file without any modifications
        df.to_csv(output_file, index=False)
        
        print(f"Processed CSV. Output saved to {output_file}")
        
        # Store the processed DataFrame in the instance
        self.processed_df = df
        return df

    def extract_mact_from_file(self, file_path, data_directory=None):
        """
        Extract Mact value from a file's metadata.
        If file_path starts with 'spectrum_X', it will look for any .pow file in the data_directory
        that contains 'spectrum_X' in its name and extract the Mact value from there.
        
        Args:
            file_path (str): Path or filename of the target file
            data_directory (str): Directory containing the data files (defaults to self.data_directory)
        
        Returns:
            float or None: Mact value as float or None if not found
        """
        if data_directory is None:
            data_directory = self.data_directory
            if data_directory is None:
                raise ValueError("No data directory specified")
        
        # Extract the base filename without directory
        base_filename = os.path.basename(file_path)
        
        # Check if the filename starts with 'spectrum_' followed by a number
        spectrum_match = re.match(r'spectrum_(\d+)', base_filename)
        
        if spectrum_match:
            # Extract the spectrum number
            spectrum_number = spectrum_match.group(1)
            
            # Find all .pow files in the data directory
            pow_files = []
            for filename in os.listdir(data_directory):
                if filename.endswith('.pow') and f'spectrum_{spectrum_number}' in filename:
                    pow_files.append(os.path.join(data_directory, filename))
            
            # If any matching .pow files were found, use the first one
            if pow_files:
                file_path = pow_files[0]
                print(f"Using metadata from {os.path.basename(file_path)} for spectrum_{spectrum_number}")
        
        # Now proceed with extracting the Mact value
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                # Search for Mact value using regex
                match = re.search(r'Mact:\s*([\d.]+)', content)
                if match:
                    return float(match.group(1))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return None

    def compare_and_plot(self, processed_df=None, data_directory=None, filename_column=None, 
                        output_plot="mass_comparison.png"):
        """
        Compare guessed mass values with actual values from metadata and create a plot.
        Plot guessed mass (y-axis) against actual mass (x-axis).
        
        Args:
            processed_df (DataFrame): DataFrame with processed mass guess values
            data_directory (str): Directory containing the actual data files
            filename_column (str): Column name in DataFrame that contains filenames
            output_plot (str): Path to save the output plot
            
        Returns:
            dict: Dictionary containing lists of guessed and actual masses
        """
        # Use instance variables if not provided
        if processed_df is None:
            processed_df = self.processed_df
            if processed_df is None:
                raise ValueError("No processed DataFrame available")
        
        if data_directory is None:
            data_directory = self.data_directory
            if data_directory is None:
                raise ValueError("No data directory specified")
                
        if filename_column is None:
            filename_column = self.filename_column
        
        if filename_column not in processed_df.columns:
            raise ValueError(f"DataFrame must contain '{filename_column}' column")
        
        # Create lists to store data for plotting
        filenames = []
        guessed_masses = []
        actual_masses = []
        
        # Process each file to extract metadata
        for _, row in enumerate(processed_df.iterrows()):
            filename = row[1][filename_column]
            guessed_mass = row[1]['mass guess']
            
            # Skip if filename is missing or mass guess is NaN
            if pd.isna(filename) or pd.isna(guessed_mass):
                continue
            
            # Extract Mact from file metadata, passing both filename and data_directory
            # This will use the updated extract_mact_from_file method to find the correct file
            mact = self.extract_mact_from_file(filename, data_directory)
            
            # If Mact was found, add to our plot data
            if mact is not None:
                filenames.append(filename)
                guessed_masses.append(guessed_mass)
                actual_masses.append(mact)
        
        # Create a DataFrame with the comparison results
        self.comparison_df = pd.DataFrame({
            'filename': filenames,
            'guessed_mass': guessed_masses,
            'actual_mass': actual_masses,
            'difference': [g - a for g, a in zip(guessed_masses, actual_masses)],
            'percent_error': [abs((g - a) / a) * 100 for g, a in zip(guessed_masses, actual_masses)]
        })
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot data with guessed mass on y-axis and actual mass on x-axis
        plt.scatter(actual_masses, guessed_masses, color='blue', alpha=0.7)
        
        # Add a reference line (y=x) to show perfect correlation
        if guessed_masses and actual_masses:  # Check if lists are not empty
            min_val = min(min(actual_masses), min(guessed_masses))
            max_val = max(max(actual_masses), max(guessed_masses))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Match Line')
        
        # Add labels and title
        plt.xlabel('Actual Mass (Mact from metadata)')
        plt.ylabel('Guessed Mass')
        plt.title('Guessed Mass vs Actual Mass Comparison')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Add text with correlation coefficient if there are enough points
        if len(actual_masses) > 1:
            correlation = np.corrcoef(actual_masses, guessed_masses)[0, 1]
            plt.annotate(f'Correlation: {correlation:.4f}', 
                        xy=(0.05, 0.95), 
                        xycoords='axes fraction', 
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Make the plot axes equal to properly show the reference line
        plt.axis('equal')
        
        # Save the plot
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {output_plot}")
        
        # Store and return the results
        self.comparison_results = {
            'filenames': filenames,
            'guessed_masses': guessed_masses,
            'actual_masses': actual_masses
        }
        return self.comparison_results



    def save_comparison_to_csv(self, output_file="mass_comparison_results.csv"):
        """
        Save the comparison results to a CSV file.
        
        Args:
            output_file (str): Path to save the CSV file
            
        Returns:
            pandas.DataFrame: The comparison DataFrame that was saved
        """
        if self.comparison_df is None or self.comparison_df.empty:
            raise ValueError("No comparison results available to save")
        
        self.comparison_df.to_csv(output_file, index=False)
        print(f"Comparison results saved to {output_file}")
        return self.comparison_df

    def print_summary_statistics(self):
        """
        Print summary statistics about the comparison results.
        
        Returns:
            dict: Dictionary containing summary statistics
        """
        if not self.comparison_results or not self.comparison_results['guessed_masses']:
            print("No comparison results available")
            return None
        
        guessed_masses = self.comparison_results['guessed_masses']
        actual_masses = self.comparison_results['actual_masses']
        guessed_avg = sum(guessed_masses) / len(guessed_masses)
        actual_avg = sum(actual_masses) / len(actual_masses)
        difference = abs(guessed_avg - actual_avg)
        
        summary = {
            "num_files": len(guessed_masses),
            "avg_guessed_mass": guessed_avg,
            "avg_actual_mass": actual_avg,
            "difference": difference,
            "correlation": np.corrcoef(actual_masses, guessed_masses)[0, 1] if len(guessed_masses) > 1 else None
        }
        
        print(f"Summary Statistics:")
        print(f"Number of files processed: {summary['num_files']}")
        print(f"Average guessed mass: {summary['avg_guessed_mass']:.4f}")
        print(f"Average actual mass: {summary['avg_actual_mass']:.4f}")
        print(f"Difference: {summary['difference']:.4f}")
        if summary['correlation'] is not None:
            print(f"Correlation coefficient: {summary['correlation']:.4f}")
        
        return summary

    def run_full_analysis(self, input_csv=None, data_directory=None, filename_column=None, 
                         output_plot="mass_comparison.png", output_csv="mass_comparison_results.csv"):
        """
        Run the complete workflow: process CSV, compare with actual masses, generate analysis,
        and save results to a CSV file.
        
        Args:
            input_csv (str): Path to the input CSV file
            data_directory (str): Directory containing the actual data files
            filename_column (str): Column name in DataFrame that contains filenames
            output_plot (str): Path to save the output plot
            output_csv (str): Path to save the comparison results CSV
            
        Returns:
            dict: Summary statistics of the analysis
        """
        # Use instance variables if not provided
        if input_csv is None:
            input_csv = self.results_csv
        if data_directory is None:
            data_directory = self.data_directory
        if filename_column is None:
            filename_column = self.filename_column
            
        # Step 1: Process the CSV to assign random mass values
        self.process_csv(input_csv)
        
        # Step 2: Extract actual masses from metadata and create comparison plot
        self.compare_and_plot(data_directory=data_directory, 
                             filename_column=filename_column, 
                             output_plot=output_plot)
        
        # Step 3: Save comparison results to CSV
        self.save_comparison_to_csv(output_csv)
        
        # Step 4: Print and return summary statistics
        return self.print_summary_statistics()
