import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from pathlib import Path
import matplotlib.backends.backend_pdf
import time
import shutil

class IterativeSNRAnalyzer:
    def __init__(self, input_directory):
        """
        Initialize the SNR analyzer with the input directory path.
        
        Parameters:
        input_directory (str): Path to the directory containing .pow files to analyze
        """
        self.input_directory = input_directory
        
        # Create output directories in the same folder as the input directory
        parent_dir = os.path.dirname(input_directory)
        self.output_directory = os.path.join(parent_dir, 'Iterative analysis output')
        self.true_directory = os.path.join(parent_dir, 'True return')
        self.false_directory = os.path.join(parent_dir, 'False return')
        self.maybe_directory = os.path.join(parent_dir, 'Maybe return')
        
        # Create the directories if they don't exist
        for directory in [self.output_directory]:
            os.makedirs(directory, exist_ok=True)
            
        # Define paths for output files
        self.pdf_path = os.path.join(self.output_directory, 'SNR_plots.pdf')
        self.csv_path = os.path.join(self.output_directory, 'Results.csv')
        
        # Storage for results
        self.results = []
        
    def process_files(self, percentlimit=14, noiselimit=10, greyarea=2, use_maybe_dir=True):
        """
        Process all .pow files in the input directory
        
        Parameters:
        percentlimit (int): Percentage limit for SNR classification (default: 14)
        noiselimit (int): Noise threshold limit in percentage (default: 10)
        greyarea (int): Grey area range for dubious detections (default: 2)
        use_maybe_dir (bool): If True, moves dubious files to Maybe directory instead of False
        """
        # Create PDF for plots
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.pdf_path)
        print(f"PDF will be saved to: {self.pdf_path}")
        
        # Get all .pow files in the directory
        pow_files = [f for f in os.listdir(self.input_directory) if f.endswith('.pow')]
        print(f"Found {len(pow_files)} .pow files in {self.input_directory}")
        
        for pow_file in pow_files:
            try:
                full_file_path = os.path.join(self.input_directory, pow_file)
                print(f"Processing: {full_file_path}")
                
                # Read and process the data
                columns = ["frequency", "power"]
                data = pd.read_csv(full_file_path, delim_whitespace=True, names=columns, skiprows=39, encoding='latin-1')
                
                numpyframe = data.to_numpy()
                frequency = (numpyframe[:, 0])
                power = numpyframe[:, 1]
                
                scalefactor = len(frequency)/842385
                cfrequency = frequency[int(10000*scalefactor):]
                cpower = power[int(10000*scalefactor):]
                
                # Create logarithmic bins
                bins333 = int(1000*scalefactor)
                logbins333 = np.logspace(np.log10(cfrequency[0]), np.log10(cfrequency[-1]), bins333)
                
                binvaluey333 = [
                    cpower[(cfrequency >= logbins333[i]) & (cfrequency < logbins333[i + 1])].mean()
                    for i in range(len(logbins333) - 1)
                ]
                
                binvaluex333 = []
                for i in range(len(logbins333)-1):
                    newx = (logbins333[i+1] + logbins333[i])/2
                    binvaluex333.append(newx)
                
                # Spreading function for non-linear binning
                length = 1
                binsss = 50
                maxfactor = 15  # max factor of multiplication
                oscpoint = 0.99  # % distance out of 0 to 1
                steep = 4  # how fast the graph falls
                
                maxf = (maxfactor-1)
                leng = np.linspace(0, length, binsss)
                
                spreading = []
                for i in range(len(leng)):
                    spreadx = (((math.exp(leng[i]**2)-1)/(math.exp(1)-1))**steep)*maxf + 1
                    spreading.append(spreadx)
                
                bins111 = binsss
                logbins111 = np.logspace(np.log10(cfrequency[0]), np.log10(cfrequency[-1]), bins111)
                logbins111end = logbins111[-1]
                
                distance1 = []
                for i in range(len(logbins111)-1):
                    jump1 = logbins111[i+1] - logbins111[i]
                    distance1.append(jump1)
                
                newlogbins111 = [logbins111[0]]
                for i in range(len(distance1)):
                    newlogbins111p = newlogbins111[i]+(distance1[i]*spreading[i])
                    newlogbins111.append(newlogbins111p)
                
                newlogbins111 = (newlogbins111 - newlogbins111[0])*logbins111end/(newlogbins111[-1]) + newlogbins111[0]
                
                binvaluey = [
                    cpower[(cfrequency >= newlogbins111[i]) & (cfrequency < newlogbins111[i + 1])].mean()
                    for i in range(len(newlogbins111) - 1)
                ]
                
                binvaluex = []
                for i in range(len(newlogbins111)-1):
                    newx = (newlogbins111[i+1] + newlogbins111[i])/2
                    binvaluex.append(newx)
                
                binvaluex333l = binvaluex333.copy()
                binvaluey333l = binvaluey333.copy()
                
                # Filter by frequency
                binrange = np.linspace(len(binvaluex333l) - 1, 0, len(binvaluex333l))
                for i in binrange:
                    if binvaluex333l[int(i)] < 800:
                        binvaluex333l = np.delete(binvaluex333l, int(i))
                        binvaluey333l = np.delete(binvaluey333l, int(i))
                
                # Smoothing iterations
                iterations = 2
                count = 0
                
                while count < iterations:  
                    binrange = np.linspace(len(binvaluex333l) - 1, 0, len(binvaluex333l))
                    
                    linefitloop = np.interp(binvaluex333l, binvaluex, binvaluey)
                    linefitloop[np.isnan(linefitloop)] = 0
                    binvaluex333lfreeze = binvaluex333l.copy()
                    
                    for i in binrange:
                        if binvaluey333l[int(i)] > linefitloop[int(i)]:
                            binvaluex333l = np.delete(binvaluex333l, int(i))
                            binvaluey333l = np.delete(binvaluey333l, int(i))
                    
                    # Remaking the smoothed
                    binvaluey = [
                        binvaluey333l[(binvaluex333l >= newlogbins111[i]) & (binvaluex333l < newlogbins111[i + 1])].mean()
                        for i in range(len(newlogbins111) - 1)
                    ]
                    
                    binvaluex = []
                    for i in range(len(newlogbins111)-1):
                        newx = (newlogbins111[i+1] + newlogbins111[i])/2
                        binvaluex.append(newx)
                        
                    count = count + 1
                
                binvaluex333f = binvaluex333.copy()
                binvaluey333f = binvaluey333.copy()
                
                # Filter data by frequency range
                sslengthfinal = np.linspace(len(binvaluex)-1, 0, len(binvaluex))
                for i in sslengthfinal:
                    if binvaluex[int(i)] < 1500 or binvaluex[int(i)] > 7000:
                        binvaluex = np.delete(binvaluex, int(i))
                        binvaluey = np.delete(binvaluey, int(i))
                
                lengthfinal = np.linspace(len(binvaluex333f)-1, 0, len(binvaluex333f))
                for i in lengthfinal:
                    if binvaluex333f[int(i)] < 1500 or binvaluex333f[int(i)] > 7000:
                        binvaluex333f = np.delete(binvaluex333f, int(i))
                        binvaluey333f = np.delete(binvaluey333f, int(i))
                
                redy = np.interp(binvaluex333f, binvaluex, binvaluey)
                
                # Calculate SNR
                SNR = []
                for i in range(len(binvaluex333f)):
                    newSNR = binvaluey333f[int(i)]/redy[int(i)]
                    SNR.append(newSNR)
                
                # Plot SNR vs Frequency
                figsnr = plt.figure(figsize=(10, 6))
                plt.plot(binvaluex333f, SNR, marker='o', color='k', label='Signal to Noise Ratio', markersize=2)
                plt.xlabel('Central Frequency (μHz)')
                plt.ylabel('Signal to Noise Ratio')
                plt.title(f'SNR vs Central Frequency for {pow_file} (1000 - 7500μhz)')
                plt.xlim(1000, 7500)
                plt.legend()
                plt.grid(alpha=0.3)
                
                pdf.savefig(figsnr)
                plt.close(figsnr)
                
                # Calculate SNR percentages
                SNRordered = np.flip(np.sort(SNR))
                percent = np.linspace(1, 100, 100)
                SNRpercent = []
                for i in percent:
                    n = int(len(SNRordered)*(i/100))
                    SNRSnip = SNRordered[:n]
                    SNRAverage = np.mean(SNRSnip)
                    SNRpercent.append(SNRAverage)
                
                # Calculate noise threshold
                SNRTotalAverage = np.mean(SNR)
                SNRTotalAverageArb = SNRTotalAverage * ((noiselimit + 100)/100)
                
                # Find percentage where SNR drops below threshold
                diff = SNRpercent - SNRTotalAverageArb
                sign_changes = None
                for i in range(len(diff)):
                    if diff[int(i)] > 0:
                        continue
                    else:
                        sign_changes = percent[int(i)]
                        break
                
                # Plot SNR percentage graph
                figper = plt.figure(figsize=(10, 6))
                plt.plot(percent, SNRpercent, marker='o', color='b', label='SNR with increasing percentage', markersize=2)
                plt.xlabel('Percent of data used')
                plt.ylabel('SNR')
                plt.legend()
                plt.title(f'SNR variation for top n% of data for {pow_file}. Drowned by noise at {sign_changes}%.')
                plt.axhline(SNRTotalAverageArb, linestyle = '-', color = 'g', label='Noise Threshold')
                plt.axvline(sign_changes, linestyle = '-', color = 'r', label='Percentile at Noise Threshold')
                plt.axvline(percentlimit, linestyle = '--', color = 'k', label='Percentile limit of falling')
                
                pdf.savefig(figper)
                plt.close(figper)
                
                # Classify detection and move file
                if sign_changes > percentlimit:
                    detection = 'True'
                    print('True detection')
                    dest_path = os.path.join(self.true_directory, pow_file)
                    # shutil.copy2(full_file_path, dest_path)
                     # Finding νmax

                    neu = []
                    for i in range(len(binvaluex333f)):
                        newneu = binvaluey333f[int(i)]-redy[int(i)]
                        neu.append(newneu)
                    neu = np.array(neu)*10000
                    figneu = plt.figure(figsize = (10, 6))
                    plt.plot(binvaluex333f, neu, marker='o', color='k', label='ν', markersize=2)
                    plt.xlabel('Central Frequency (μHz)')
                    plt.ylabel('ν (μHz)')
                    plt.title(f'ν vs Central Frequency for Binned Data (1000 - 7500μhz)')
                    plt.xlim(500, 8000)
                    plt.legend()
                    plt.grid(alpha=0.3)
                    pdf.savefig(figneu)
                    plt.close(figneu)
                    # Finding νmax

                    percentile = 2 # anything higher starts to cause some issues

                    cut = int((percentile/100)*len(neu))

                    neuorder = np.argsort(neu)
                    sortedneu = neu[neuorder[::-1][:cut]]
                    sortedfreq = binvaluex333f[neuorder[::-1][:cut]]

                    sortedsortedfreq = np.sort(sortedfreq)
                    ssfds = []

                    sortedrange = np.linspace(0, len(sortedsortedfreq)-2, len(sortedsortedfreq)-1)
                    for i in sortedrange:
                        ssfds.append(sortedsortedfreq[int(i+1)]-sortedsortedfreq[int(i)])

                    maxsplit = max(ssfds)
                    if maxsplit > 1000:
                        splitpeakA = sortedsortedfreq[:ssfds.index(maxsplit)+1]
                        splitpeakB = sortedsortedfreq[ssfds.index(maxsplit)+1:]
                        thenewneus = []
                        
                        sortedfreq = sortedfreq.tolist()
                        for i in splitpeakB:
                            thenewneus.append(sortedneu[sortedfreq.index(i)])
                        sortedneu = thenewneus
                        sortedfreq = splitpeakB

                    neumax = np.mean(sortedfreq)
                    neuerror = np.std(sortedfreq)

                    rounding = 3
                    neumax = np.round(neumax, rounding)
                    neuerror = np.round(neuerror, rounding)

                    figtopneu = plt.figure(figsize=(10, 6))
                    plt.scatter(sortedfreq, sortedneu, marker='o', color='k', label='ν')
                    plt.errorbar(neumax, np.mean(sortedneu), xerr=neuerror, marker='o', color='b', label=f'νmax = {neumax}μhz ± {neuerror}μhz')
                    plt.xlabel('Central Frequency (μHz)')
                    plt.ylabel('ν (μHz)')
                    plt.title(f'Top {percentile}% ν vs Central Frequency for Binned Data (1000 - 7500μhz)')
                    plt.xlim(500, 8000)
                    plt.ylim(0, max(sortedneu)*1.1)
                    plt.legend()
                    plt.grid(alpha=0.3)
                    pdf.savefig(figtopneu)
                    plt.close(figtopneu)
                    print(f'νmax = {neumax}μhz ± {neuerror}μhz to {rounding}dp at the top {percentile}%')


                elif sign_changes >= percentlimit - greyarea and sign_changes <= percentlimit:
                    detection = 'MAYBE'
                    print('Detection dubious')
                    neumax = ''
                    neuerror = ''
                    if use_maybe_dir:
                        dest_path = os.path.join(self.maybe_directory, pow_file)
                    else:
                        dest_path = os.path.join(self.false_directory, pow_file)
                    # shutil.copy2(full_file_path, dest_path)
                else:
                    detection = 'False'
                    print('False detection')
                    neumax = ''
                    neuerror = ''
                    dest_path = os.path.join(self.false_directory, pow_file)
                    # shutil.copy2(full_file_path, dest_path)
                
                # Store the results
                self.results.append({
                    'filename': pow_file,
                    'snr': detection,
                    'gaussian': '',  # Added blank column L
                    'spacings': '',  # Added blank column F
                    'nu_max': neumax,  # Added blank column Nu_max
                    'error nu_max': neuerror,  # Added blank column Error Nu_max
                    'delta nu': '',  # Added blank column Delta Nu
                    'error delta nu': '',  # Added blank column Error Delta Nu
                    'mass guess': '', #Blank col for mass guess.
                    'mass error': '', #Blank col for mass error.
                    'radius guess': '', #Blank col for rad guess.
                    'radius error': '' #Blank col for rad error.
                })
                
            except Exception as e:
                print(f"Error processing {pow_file}: {str(e)}")
                self.results.append({
                    'filename': pow_file,
                    'full_path': os.path.join(self.input_directory, pow_file),
                    'Noise_Drown': None,
                    'detection': 'ERROR',
                    'L': '',  # Added blank column L
                    'F': '',  # Added blank column F
                    'Nu_max': 'ERROR',  # Added blank column Nu_max
                    'Error Nu_max': 'ERROR',  # Added blank column Error Nu_max
                    'Delta Nu': '',  # Added blank column Delta Nu
                    'Error Delta Nu': '',  # Added blank column Error Delta Nu
                    'mass guess': '', #Blank col for mass guess.
                    'radius guess': '' #Blank col for rad guess.
                })
        
        # Close the PDF file
        pdf.close()
        print(f"PDF file saved: {self.pdf_path}")
        
        # Save results to CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.csv_path, index=False)
        print(f"CSV results file saved: {self.csv_path}")
        
        # Print summary
        print("\nResults Summary:")
        print(results_df.head())
        print(f"Total files processed: {len(results_df)}")
        print(f"All plots saved to {self.output_directory}")
        
        return results_df