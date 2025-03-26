import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.optimize import curve_fit # type: ignore
from scipy.signal import find_peaks # type: ignore

class GaussianWrapper:
    '''
    This can be run as following:
    gw = GaussianWrapper(filepath)
    status, nu_max, envelope_indices, best_fit_no = gw.run_wrapper()

    Where filepath is a file containing the best three filters.

    This will output a tuple with three values:
    1. 'True' if a Gaussian is spotted, 'False' otherwise.
    2. The nu_max and its uncertainty of the best-fitting Gaussian.
    3. The indices of the Gaussian peak within the data for each filter (if detected)

    The filepath should link to a CSV with the first column containing frequencies 
    and the second to fourth columns containing data points.
    '''
    def __init__(self, csv_path):
        """
        Initialize the GaussianWrapper class with data from a CSV file.
        
        Parameters:
        csv_path (str): The file path to the CSV file containing the data.
        """
        self.df = pd.read_csv(csv_path)
        self.freq = self.df[self.df.columns[0]]
        self.data = self.df[self.df.columns[1:4]]
        self.gf_list = []
        
        self.fit = None
        self.nu_max = []
        self.gauss_indices = []
        self.spotted = False
        self.best_fit_no = None
    
    def run_wrapper(self):
        """
        Executes the fitting and parameter checking for multiple Gaussian fitters.
        Iterates over a list of Gaussian fitter objects, performing fitting and parameter
        checks. Updates the attributes `spotted`, `goodness`, `nu_max`, and `peak_indices`
        based on the results of the fitters.
        Returns:
            tuple: A tuple containing:
                - spotted (bool): Indicates if a Gaussian was spotted.
                - nu_max (list): The peak position and its uncertainty.
                - gauss_indices (numpy.ndarray): Indices of the Gaussian peak within the data for each filter.
        """
        col_no = 1
        for column in self.data.columns:
            gf = GaussianFitter(self.freq, self.df[column])
            gf.fit()
            gf.parameter_check()
            self.gf_list.append(gf)
            if gf.gauss_spotted:
                self.spotted = True
                if self.fit is None or gf.goodness < self.fit:
                    self.fit = gf.goodness
                    self.nu_max = [gf.popt[1], np.sqrt(np.diag(gf.pcov))[1]]
                    index_range = np.where(gf.ym > np.min(gf.ym) + 1)[0]
                    self.best_fit = col_no
                self.gauss_indices.append([min(index_range), max(index_range)])
            else:
                self.gauss_indices.append([min(self.df.index), max(self.df.index)])
            col_no += 1
                
        
        return self.spotted, self.nu_max, self.gauss_indices, self.best_fit_no

class GaussianFitter:
    def __init__(self, freq, data, distance=5000):
        """
        Initialize the GaussianFitter instance.

        Parameters:
        freq (array-like): Array of frequency values. Should be of the same shape as `data`.
        data (array-like): Array of data values corresponding to the frequencies in `freq`.
        distance (int, optional): Minimum distance between detected peaks. Default is 10000.
        """
        self.freq = freq
        self.data = data  # Store dataset for this instance
        self.distance = distance  # Peak detection distance
        self.min_height = None
        self.popt = None  # Optimal parameters from curve fitting
        self.pcov = None  # Covariance of the parameters
        self.goodness = None  # Goodness-of-fit measure
        self.ym = None  # Fitted Gaussian values
        self.peaks = None  # Detected peaks
        self.fwhm = None # Full-width at half maximum
        self.gauss_spotted = None # Gaussian spotted or not

    @staticmethod    
    def gaussian(x, a, x0, sigma):
        """
        Compute the value of a Gaussian function.
        Parameters:
        x (float or array-like): The input value(s) where the Gaussian function is evaluated.
        a (float): The amplitude of the Gaussian function.
        x0 (float): The mean or the center of the Gaussian peak.
        sigma (float): The standard deviation, which controls the width of the Gaussian peak.
        Returns:
        float or array-like: The value of the Gaussian function at the given input value(s).
        """

        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    def fit(self):
        """
        Fits a Gaussian function to the data stored in the instance.
        Attributes:
            peaks (ndarray): Indices of the peaks found in the data.
            gauss_spotted (bool): Indicates whether a Gaussian fit was successfully performed.
            min_height (float): Minimum height threshold for filtering peaks.
            popt (ndarray): Optimal values for the parameters of the Gaussian function.
            pcov (ndarray): Covariance of the parameters.
            ym (ndarray): Fitted Gaussian values.
            goodness (float): Measure of the goodness-of-fit.
        Raises:
            Exception: If an error occurs during the fitting process, sets `gauss_spotted` to False.
        """

        try:
            # Find peaks in the stored data
            self.peaks, _ = find_peaks(self.data, distance=self.distance)
            
            # Fit Gaussian if peaks exist
            if len(self.peaks) == 0:
                self.gauss_spotted = False
            
            # Removing low outliers and bringing the base of the Gaussian to zero
            std_dev = np.std(self.data[self.peaks])
            self.min_height = np.max(self.data) - 3 * std_dev
            mask = self.data[self.peaks] >= self.min_height
            
            filt_freq = self.freq[self.peaks][mask]
            filt_data = self.data[self.peaks][mask] - self.min_height
                
            # Optimisation
            initial_guess = [np.max(filt_data), filt_freq[filt_data.idxmax()], np.std(filt_freq)]
            self.popt, self.pcov = curve_fit(self.gaussian, filt_freq, filt_data, p0=initial_guess)
            perr = np.sqrt(np.diag(self.pcov)) # Standard deviation of the parameters
            self.ym = self.gaussian(self.freq, *self.popt) # Fitted Gaussian values
            self.goodness = np.sum(abs(perr) / abs(self.popt)) 
            
        except:
            self.gauss_spotted = False 
    
    def parameter_check(self):
        """
        Checks the parameters of the Gaussian fit to determine if a valid Gaussian has been spotted.
        Returns:
            bool: True if a valid Gaussian is spotted based on the checks, False otherwise.
        """

        try:
            fwhm_points = np.where(self.ym > np.max(self.ym)/2)[0]
            self.fwhm = self.freq[np.max(fwhm_points)] - self.freq[np.min(fwhm_points)] 
            g_check = True if self.goodness is not None and abs(self.goodness) < 0.3 else False 
            fwhm_check = True if 500 < self.fwhm < 2500 else False 
            x0_check = True if 1500 < self.popt[1] < 5700 else False 
            self.gauss_spotted = True if g_check and fwhm_check and x0_check else False 
            self.ym += self.min_height 
        
        except:
            self.gauss_spotted = False
           
        return self.gauss_spotted