import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re


class StellarParameterComparison:
    """
    A class for comparing guessed stellar parameters (mass or radius) with actual values from metadata files,
    processing CSV files, and generating visualizations of the results.
    """

    def __init__(self, results_csv=None, data_directory=None, filename_column="filename", 
                 parameter_type="mass"):
        """
        Initialize the StellarParameterComparison object.

        Args:
            results_csv (str): Path to the CSV file containing guessed parameters
            data_directory (str): Directory containing the actual data files with metadata
            filename_column (str): Name of the column in CSV that contains filenames
            parameter_type (str): Type of parameter to compare ("mass" or "radius")
        """
        self.results_csv = results_csv
        self.data_directory = data_directory
        self.filename_column = filename_column
        self.parameter_type = parameter_type.lower()
        
        # Set parameter-specific attributes based on parameter_type
        if self.parameter_type == "mass":
            self.metadata_pattern = r'Mact:\s*([\d.]+)'
            self.guess_column = 'mass guess'
            self.parameter_name = 'Mass'
        elif self.parameter_type == "radius":
            self.metadata_pattern = r'Radius:\s*([\d.]+)'
            self.guess_column = 'radius guess'
            self.parameter_name = 'Radius'
        else:
            raise ValueError("Parameter type must be 'mass' or 'radius'")
            
        self.processed_df = None
        self.comparison_results = None
        self.comparison_df = None

    def process_csv(self, input_file=None, output_file=None):
        """
        Process a CSV file without modifying any parameter values.
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
        if self.guess_column not in df.columns:
            raise ValueError(f"CSV must contain '{self.guess_column}' column")
        
        # Save the dataframe to a new CSV file without any modifications
        df.to_csv(output_file, index=False)
        
        print(f"Processed CSV. Output saved to {output_file}")
        
        # Store the processed DataFrame in the instance
        self.processed_df = df
        return df

    def extract_parameter_from_file(self, file_path, data_directory=None):
        """
        Extract parameter value (mass or radius) from a file's metadata.
        If file_path starts with 'spectrum_X', it will look for any .pow file in the data_directory
        that contains 'spectrum_X' in its name and extract the parameter value from there.
        
        Args:
            file_path (str): Path or filename of the target file
            data_directory (str): Directory containing the data files (defaults to self.data_directory)
        
        Returns:
            float or None: Parameter value as float or None if not found
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
        
        # Now proceed with extracting the parameter value
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                # Search for parameter value using regex
                match = re.search(self.metadata_pattern, content)
                if match:
                    return float(match.group(1))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return None

    def compare_and_plot(self, processed_df=None, data_directory=None, filename_column=None, 
                         output_plot=None):
        """
        Compare guessed parameter values with actual values from metadata and create a plot.
        Plot guessed parameter (y-axis) against actual parameter (x-axis).
        
        Args:
            processed_df (DataFrame): DataFrame with processed parameter guess values
            data_directory (str): Directory containing the actual data files
            filename_column (str): Column name in DataFrame that contains filenames
            output_plot (str): Path to save the output plot
            
        Returns:
            dict: Dictionary containing lists of guessed and actual parameter values
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
        
        if output_plot is None:
            output_plot = f"{self.parameter_type}_comparison.png"
        
        if filename_column not in processed_df.columns:
            raise ValueError(f"DataFrame must contain '{filename_column}' column")
        
        # Create lists to store data for plotting
        filenames = []
        guessed_values = []
        actual_values = []
        
        # Process each file to extract metadata
        for _, row in enumerate(processed_df.iterrows()):
            filename = row[1][filename_column]
            guessed_value = row[1][self.guess_column]
            
            # Skip if filename is missing or parameter guess is NaN
            if pd.isna(filename) or pd.isna(guessed_value):
                continue
            
            # Extract parameter from file metadata, passing both filename and data_directory
            # This will use the updated extract_parameter_from_file method to find the correct file
            actual_value = self.extract_parameter_from_file(filename, data_directory)
            
            # If parameter was found, add to our plot data
            if actual_value is not None:
                filenames.append(filename)
                guessed_values.append(guessed_value)
                actual_values.append(actual_value)
        
        # Create a DataFrame with the comparison results
        self.comparison_df = pd.DataFrame({
            'filename': filenames,
            f'guessed_{self.parameter_type}': guessed_values,
            f'actual_{self.parameter_type}': actual_values,
            'difference': [g - a for g, a in zip(guessed_values, actual_values)],
            'percent_error': [abs((g - a) / a) * 100 for g, a in zip(guessed_values, actual_values)]
        })
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot data with guessed parameter on y-axis and actual parameter on x-axis
        plt.scatter(actual_values, guessed_values, color='blue', alpha=0.7)
        
        # Add a reference line (y=x) to show perfect correlation
        if guessed_values and actual_values:  # Check if lists are not empty
            min_val = min(min(actual_values), min(guessed_values))
            max_val = max(max(actual_values), max(guessed_values))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Match Line')
        
        # Add labels and title
        plt.xlabel(f'Actual {self.parameter_name} ({self.parameter_name} from metadata)')
        plt.ylabel(f'Guessed {self.parameter_name}')
        plt.title(f'Guessed {self.parameter_name} vs Actual {self.parameter_name} Comparison')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Add text with correlation coefficient if there are enough points
        if len(actual_values) > 1:
            correlation = np.corrcoef(actual_values, guessed_values)[0, 1]
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
            f'guessed_{self.parameter_type}s': guessed_values,
            f'actual_{self.parameter_type}s': actual_values
        }
        return self.comparison_results

    def save_comparison_to_csv(self, output_file=None):
        """
        Save the comparison results to a CSV file.
        
        Args:
            output_file (str): Path to save the CSV file
            
        Returns:
            pandas.DataFrame: The comparison DataFrame that was saved
        """
        if self.comparison_df is None or self.comparison_df.empty:
            raise ValueError("No comparison results available to save")
            
        if output_file is None:
            output_file = f"{self.parameter_type}_comparison_results.csv"
        
        self.comparison_df.to_csv(output_file, index=False)
        print(f"Comparison results saved to {output_file}")
        return self.comparison_df

    def print_summary_statistics(self):
        """
        Print summary statistics about the comparison results.
        
        Returns:
            dict: Dictionary containing summary statistics
        """
        if not self.comparison_results:
            print("No comparison results available")
            return None
            
        param_type = self.parameter_type
        guessed_key = f'guessed_{param_type}s'
        actual_key = f'actual_{param_type}s'
        
        if not self.comparison_results.get(guessed_key):
            print("No comparison results available")
            return None
        
        guessed_values = self.comparison_results[guessed_key]
        actual_values = self.comparison_results[actual_key]
        guessed_avg = sum(guessed_values) / len(guessed_values)
        actual_avg = sum(actual_values) / len(actual_values)
        difference = abs(guessed_avg - actual_avg)
        
        summary = {
            "num_files": len(guessed_values),
            f"avg_guessed_{param_type}": guessed_avg,
            f"avg_actual_{param_type}": actual_avg,
            "difference": difference,
            "correlation": np.corrcoef(actual_values, guessed_values)[0, 1] if len(guessed_values) > 1 else None
        }
        
        print(f"Summary Statistics for {self.parameter_name}:")
        print(f"Number of files processed: {summary['num_files']}")
        print(f"Average guessed {param_type}: {summary[f'avg_guessed_{param_type}']:.4f}")
        print(f"Average actual {param_type}: {summary[f'avg_actual_{param_type}']:.4f}")
        print(f"Difference: {summary['difference']:.4f}")
        if summary['correlation'] is not None:
            print(f"Correlation coefficient: {summary['correlation']:.4f}")
        
        return summary

    def run_full_analysis(self, input_csv=None, data_directory=None, filename_column=None, 
                         output_plot=None, output_csv=None):
        """
        Run the complete workflow: process CSV, compare with actual parameter values, generate analysis,
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
        if output_plot is None:
            output_plot = f"{self.parameter_type}_comparison.png"
        if output_csv is None:
            output_csv = f"{self.parameter_type}_comparison_results.csv"
            
        # Step 1: Process the CSV (no modifications)
        self.process_csv(input_csv)
        
        # Step 2: Extract actual parameters from metadata and create comparison plot
        self.compare_and_plot(data_directory=data_directory, 
                             filename_column=filename_column, 
                             output_plot=output_plot)
        
        # Step 3: Save comparison results to CSV
        self.save_comparison_to_csv(output_csv)
        
        # Step 4: Print and return summary statistics
        return self.print_summary_statistics()

# For backward compatibility, create a MassComparison class that inherits from StellarParameterComparison
class MassComparison(StellarParameterComparison):
    """
    A class for comparing guessed stellar masses with actual masses from metadata files.
    This is a wrapper around StellarParameterComparison for backward compatibility.
    """
    
    def __init__(self, results_csv=None, data_directory=None, filename_column="filename"):
        """
        Initialize the MassComparison object.

        Args:
            results_csv (str): Path to the CSV file containing guessed masses
            data_directory (str): Directory containing the actual data files with metadata
            filename_column (str): Name of the column in CSV that contains filenames
        """
        super().__init__(results_csv, data_directory, filename_column, parameter_type="mass")


# Create a RadiusComparison class that inherits from StellarParameterComparison
class RadiusComparison(StellarParameterComparison):
    """
    A class for comparing guessed stellar radii with actual radii from metadata files.
    This is a wrapper around StellarParameterComparison for ease of use.
    """
    
    def __init__(self, results_csv=None, data_directory=None, filename_column="filename"):
        """
        Initialize the RadiusComparison object.

        Args:
            results_csv (str): Path to the CSV file containing guessed radii
            data_directory (str): Directory containing the actual data files with metadata
            filename_column (str): Name of the column in CSV that contains filenames
        """
        super().__init__(results_csv, data_directory, filename_column, parameter_type="radius")
