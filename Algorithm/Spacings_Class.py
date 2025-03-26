import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class Spacings:
    '''Class for my functions'''
    
    def __init__(self, filename, index, best_fit_no):
        '''
        Parameters:
        filename: the csv file containing the best three matched filter outputs and 
                  a frequency column
        index: numpy array containing the indices of the start and end of the oscillation 
               part of the spectrum for each column
        best_fit_no: column number with the best fit from the GaussianFitter class
        '''
        # Read the file
        data = pd.read_csv(filename)
        # Get the frequency column
        self.frequency0 = data[data.columns[0]].values
#         data2 = data[data.columns[2]].values
        if best_fit_no == None: # No oscillations found by Gaussian
            # Iterate as before
            # Specify the spacings using string slicing on the column names
            data1 = data[data.columns[1]].values
            data2 = data[data.columns[2]].values
            data3 = data[data.columns[3]].values
            # Have as a list to iterate over
            # No indices
            self.frequency = [self.frequency0, self.frequency0, self.frequency0]
            self.data = [data1, data2, data3]
            # Specify the spacings using string slicing on the column names
            spacing1 = float(data.columns[1][:5])
            spacing2 = float(data.columns[2][:5])
            spacing3 = float(data.columns[3][:5])
            self.spacing = [spacing1, spacing2, spacing3]
        else: # Gaussian found oscillations
            self.data = data[data.columns[best_fit_no]].values
            # NEED TO DEFINE BEST AS BEST COLUMN
            self.spacing = float(data.columns[best_fit_no][:5])
            self.frequency = [self.frequency0[index[best_fit_no-1][0]:index[best_fit_no-1][1]]]

        # Other parameters
        self.HWHM = 10 # Arbitrary
        # Set results to None for now
        self.result = None
        self.delta_nu = None
        self.delta_nu_err = None

    def spacings(self):
        '''
        Function to find whether there are oscillations in the data by seeing if there
        is a peak either side of the maximum value, at a distance of approximately the spacing.
        If there are oscillations, it then estimates the large frequency separation using 
        scipy.signal.find_peaks. 
        
        Returns:
        --------
        result: boolean, True if oscillations detected, False if not
        Delta_nu: median of the large frequency spacing for output
                  if oscillations were detected
        Delta_nu_err: median absolute error on Delta_nu
        '''
        # First check if we're using just the best set of data (numpy.ndarray) or all (list)
        # Look at best set of data if it's there
        if type(self.data) == np.ndarray:
            try:
                freq = self.frequency
                data = self.data
                # Find maximum peak in data and location
                max_power = np.max(data)
                max_power_i = np.argmax(data)

                # Frequency conversion
                # They will all have the same frequency to index conversion, due to the number of points in the dataframe
                f_range = self.frequency0[len(self.frequency0) - 1] - self.frequency0[0] # Range of frequencies
                f_to_i = int(len(self.frequency0) / f_range)   # How many indices per unit frequency, as an integer
                spacingi = int(self.spacing * f_to_i)       # Spacing converted to indices
                HWHMi = self.HWHM * f_to_i

                # Define the ranges of indices in which to search for maximums
                lower = max_power_i - spacingi - 2*HWHMi
                upper = max_power_i - spacingi + 2*HWHMi
                m = np.max(data[lower:upper])     # Find the maximum value in this range
                m_i = np.argmax(data[lower:upper])   # Index that this maximum occurs at
                lower2 = int(max_power_i+spacingi-2*HWHMi)        
                upper2 = int(max_power_i + spacingi + 2*HWHMi)
                m2 = np.max(data[lower2:upper2])     
                m_i2 = np.argmax(data[lower2:upper2])

                # Checks for false positives
                if m < 4*max_power/5:    # Check that the next peak is tall enough
                    self.result = False
                    self.delta_nu = None
                    self.delta_nu_err = None
                elif m2 < 4*max_power/5:
                    self.result = False
                    self.delta_nu = None
                    self.delta_nu_err = None
                else:
                    # Then estimate spacings
                    threshold = 2*max_power/3 # Increased the threshold from the other function
                                            # to be able to cope with noisier data
                    # Choose distance based on arbitrary lower limit on delta_nu/2
                    distance_btw_peaks = 40 * f_to_i # 40 micro Hz

                    # Find the peaks
                    peaks, height = find_peaks(self.data, threshold, distance=distance_btw_peaks) 
                    Heights = height['peak_heights']

                    # Now find the mean spacing
                    peak_spacings = []
                    for n in range (1, len(peaks)):
                        space = peaks[n] - peaks[n-1]
                        peak_spacings.append(space)
                    median_spacing = np.median(peak_spacings)

                    # Convert to micro Hz, remember to change from delta_nu/2 to delta_nu
                    # This accounts for the smaller modes
                    delta_nu = median_spacing * 2 / f_to_i
                    # Median absolute error
                    # Convert to large frequency spacings as frequency
                    spacings_freq = np.array(peak_spacings) * 2 /f_to_i
                    # Difference between each and the matched filter spacing
                    # y_predicted - y_true
                    difference = [self.spacing[i] - s for s in spacings_freq]
                    # Choose the median of these
                    delta_nu_err = np.median(np.abs(difference))

                    # Add check to make sure delta_nu is close to matched filter spacing
                    # Arbitrary tolerance of 10 microHz
                    if np.isclose(delta_nu, self.spacing, atol=10) == True:
                        # Add the results
                        self.result = True
                        self.delta_nu = delta_nu
                        self.delta_nu_err = delta_nu_err
                    else:
                        # Don't add the results if the spacing is wrong
                        self.result = False
                        self.delta_nu = None
                        self.delta_nu_err = None
            except Exception as e:
            # Set result to false, delta nu to none and delta nu err to none
                self.result = False
                self.delta_nu = None
                self.delta_nu_err = None
        
        # Iterate over list if not
        else: 
            Delta_nu = []
            Delta_nu_err = []
            result = []
            for i in range(0, 3):
                try:
                    freq = self.frequency[i]
                    data = self.data[i]
                    # Find maximum peak in data and location
                    max_power = np.max(data)
                    max_power_i = np.argmax(data)

                    # Frequency conversion
                    # They will all have the same frequency to index conversion, due to the number of points in the dataframe
                    f_range = self.frequency0[len(self.frequency0) - 1] - self.frequency0[0] # Range of frequencies
                    f_to_i = int(len(self.frequency0) / f_range)   # How many indices per unit frequency, as an integer
                    spacingi = int(self.spacing[i] * f_to_i)       # Spacing converted to indices
                    HWHMi = self.HWHM * f_to_i

                    # Define the ranges of indices in which to search for maximums
                    lower = max_power_i - spacingi - 2*HWHMi
                    upper = max_power_i - spacingi + 2*HWHMi
                    m = np.max(data[lower:upper])     # Find the maximum value in this range
                    m_i = np.argmax(data[lower:upper])   # Index that this maximum occurs at
                    lower2 = int(max_power_i+spacingi-2*HWHMi)        
                    upper2 = int(max_power_i + spacingi + 2*HWHMi)
                    m2 = np.max(data[lower2:upper2])     
                    m_i2 = np.argmax(data[lower2:upper2])

                    # Checks for false positives
                    if m < 4*max_power/5:    # Check that the next peak is tall enough
                        result.append(False)
                        Delta_nu.append(None)
                        Delta_nu_err.append(None)
                    elif m2 < 4*max_power/5:
                        result.append(False)
                        Delta_nu.append(None)
                        Delta_nu_err.append(None)
                    else:
                        # Then estimate spacings
                        threshold = 2*max_power/3 # Increased the threshold from the other function
                                                # to be able to cope with noisier data
                        # Choose distance based on arbitrary lower limit on delta_nu/2
                        distance_btw_peaks = 40 * f_to_i # 40 micro Hz

                        # Find the peaks
                        peaks, height = find_peaks(self.data[i], threshold, distance=distance_btw_peaks) 
                        Heights = height['peak_heights']

                        # Now find the mean spacing
                        peak_spacings = []
                        for n in range (1, len(peaks)):
                            space = peaks[n] - peaks[n-1]
                            peak_spacings.append(space)
                        median_spacing = np.median(peak_spacings)

                        # Convert to micro Hz, remember to change from delta_nu/2 to delta_nu
                        # This accounts for the smaller modes
                        delta_nu = median_spacing * 2 / f_to_i
                        # Median absolute error
                        # Convert to large frequency spacings as frequency
                        spacings_freq = np.array(peak_spacings) * 2 /f_to_i
                        # Difference between each and the matched filter spacing
                        # y_predicted - y_true
                        difference = [self.spacing[i] - s for s in spacings_freq]
                        # Choose the median of these
                        delta_nu_err = np.median(np.abs(difference))

                        # Add check to make sure delta_nu is close to matched filter spacing
                        # Arbitrary tolerance of 10 microHz
                        if np.isclose(delta_nu, self.spacing[i], atol=10) == True:
                            # Add the results
                            result.append(True)
                            Delta_nu.append(delta_nu)
                            Delta_nu_err.append(delta_nu_err)
                        else:
                            # Don't add the results if the spacing is wrong
                            result.append(False)
                            Delta_nu.append(None)
                            Delta_nu_err.append(None)
                except Exception as e:
                # Set result to false, delta nu to none and delta nu err to none
                    result.append(False)
                    Delta_nu.append(None)
                    Delta_nu_err.append(None)

                # Find best result to return
                # First check in case no oscillations were detected
                if np.any(result) == False: # If there were no True results
                    self.result = False
                else:
                    # Choose the result with the smallest error
                    chosen = np.nanargmin(np.array(Delta_nu_err, dtype=float))
                    # Need to use this code to change None to nan so that nanargmin will behave
                    self.result = result[chosen]
                    self.delta_nu = Delta_nu[chosen]
                    self.delta_nu_err = Delta_nu_err[chosen]

        return self.result, self.delta_nu, self.delta_nu_err
