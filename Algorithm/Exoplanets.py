from __future__ import print_function, division
import numpy as np
from PyAstronomy import pyasl
import matplotlib.pyplot as plt

class ExoplanetAnalysis:
    class AsteroseismologyScaling:
        """
        A class to compute stellar mass and radius using asteroseismic scaling relations, including error propagation.
        """
        # Solar reference values
        V_MAX_SOLAR = 3090  # uHz
        DELTA_V_SOLAR = 135.1  # uHz
        T_EFF_SOLAR = 5772  # K
        M_SOLAR = 2e30  # kg
        R_SOLAR = 696000  # km
    
        # Solar uncertainties
        v_max_solar_err = 30  # uHz
        delta_v_solar_err = 0.1  # uHz
        T_eff_solar_err = 0.8  # K
        R_solar_err = 1000  # km
        M_solar_err = 0.05e30  # kg (estimated uncertainty, update as needed)
    
        def __init__(self, v_max, delta_v, bv, feh, v_max_err, delta_v_err):
            """
            Initialize with observed stellar parameters.
            
            :param v_max: Frequency of maximum power (uHz)
            :param delta_v: Large frequency separation (uHz)
            :param bv: B-V color index
            :param feh: Metallicity [Fe/H]
            :param v_max_err: Uncertainty in v_max (uHz)
            :param delta_v_err: Uncertainty in delta_v (uHz)
            """
            self.v_max = v_max
            self.delta_v = delta_v
            self.bv = bv
            self.feh = feh
            self.v_max_err = v_max_err
            self.delta_v_err = delta_v_err
            self.T_eff = self._calculate_teff()
    
        def _calculate_teff(self):
            """Convert B-V color index and metallicity to effective temperature."""
            r = pyasl.Ramirez2005()
            return r.colorToTeff("B-V", self.bv, self.feh)
        
        def compute_mass(self):
            """Compute the stellar mass and its uncertainty using asteroseismic scaling relations."""
            M = (self.M_SOLAR * (self.v_max / self.V_MAX_SOLAR) ** 3 *
                 (self.delta_v / self.DELTA_V_SOLAR) ** (-4) *
                 (self.T_eff / self.T_EFF_SOLAR) ** (3/2))
    
            # Partial derivatives for uncertainty propagation
            dM_M_solar = ((self.v_max / self.V_MAX_SOLAR) ** 3 *
                          (self.delta_v / self.DELTA_V_SOLAR) ** (-4) *
                          (self.T_eff / self.T_EFF_SOLAR) ** (3/2))
    
            dM_v_max = (3 * self.M_SOLAR * self.v_max**2 * self.V_MAX_SOLAR**-3 *
                        (self.delta_v / self.DELTA_V_SOLAR) ** (-4) *
                        (self.T_eff / self.T_EFF_SOLAR) ** (3/2))
    
            dM_v_max_solar = (-3 * self.M_SOLAR * self.V_MAX_SOLAR**-4 * self.v_max**3 *
                              (self.delta_v / self.DELTA_V_SOLAR) ** (-4) *
                              (self.T_eff / self.T_EFF_SOLAR) ** (3/2))
    
            dM_delta_v = (-4 * self.M_SOLAR * self.delta_v**-5 * self.DELTA_V_SOLAR**4 *
                          (self.v_max / self.V_MAX_SOLAR) ** 3 *
                          (self.T_eff / self.T_EFF_SOLAR) ** (3/2))
    
            dM_delta_v_solar = (4 * self.M_SOLAR * self.delta_v**-4 * self.DELTA_V_SOLAR**3 *
                                (self.v_max / self.V_MAX_SOLAR) ** 3 *
                                (self.T_eff / self.T_EFF_SOLAR) ** (3/2))
    
            dM_T_eff_solar = (-3/2 * self.M_SOLAR * self.T_eff**-5/2 * self.T_EFF_SOLAR**3/2 *
                              (self.v_max / self.V_MAX_SOLAR) ** 3 *
                              (self.delta_v / self.DELTA_V_SOLAR) ** (-4))
    
            # Compute the uncertainty in M
            M_err = np.sqrt((dM_M_solar * self.M_solar_err) ** 2 +
                            (dM_v_max * self.v_max_err) ** 2 +
                            (dM_v_max_solar * self.v_max_solar_err) ** 2 +
                            (dM_delta_v * self.delta_v_err) ** 2 +
                            (dM_delta_v_solar * self.delta_v_solar_err) ** 2 +
                            (dM_T_eff_solar * self.T_eff_solar_err) ** 2)
    
            return M, M_err
    
        def compute_radius(self):
            """Compute the stellar radius and its uncertainty using asteroseismic scaling relations."""
            R = (self.R_SOLAR * (self.v_max / self.V_MAX_SOLAR) *
                 (self.delta_v / self.DELTA_V_SOLAR) ** -2 *
                 (self.T_eff / self.T_EFF_SOLAR) ** (1/2))
    
            # Partial derivatives for uncertainty propagation
            dR_v_max_solar = -(self.DELTA_V_SOLAR**2 * self.R_SOLAR * (self.T_eff / self.T_EFF_SOLAR)**(0.5) * self.v_max) / (self.delta_v**2 * self.V_MAX_SOLAR**2)
            dR_delta_v_solar = (2 * self.R_SOLAR * (self.T_eff / self.T_EFF_SOLAR)**(0.5) * self.v_max * self.DELTA_V_SOLAR) / (self.V_MAX_SOLAR * self.delta_v**2)
            dR_T_eff_solar = -(self.DELTA_V_SOLAR**2 * self.R_SOLAR * self.T_EFF_SOLAR * self.v_max) / (2 * self.V_MAX_SOLAR * self.delta_v**2 * (self.T_eff / self.T_EFF_SOLAR)**(0.5) * self.T_EFF_SOLAR**2)
            dR_v_max = (self.DELTA_V_SOLAR**2 * self.R_SOLAR * (self.T_eff / self.T_EFF_SOLAR)**(0.5)) / (self.V_MAX_SOLAR * self.delta_v**2)
            dR_delta_v = -(2 * self.DELTA_V_SOLAR**2 * self.R_SOLAR * (self.T_eff / self.T_EFF_SOLAR)**(0.5) * self.v_max) / (self.V_MAX_SOLAR * self.delta_v**3)
            dR_R_solar = (self.DELTA_V_SOLAR**2 * (self.T_eff / self.T_EFF_SOLAR)**(0.5) * self.v_max) / (self.V_MAX_SOLAR * self.delta_v**2)
    
            # Compute the uncertainty in R
            R_err = np.sqrt((dR_v_max_solar * self.v_max_solar_err) ** 2 +
                            (dR_delta_v_solar * self.delta_v_solar_err) ** 2 +
                            (dR_T_eff_solar * self.T_eff_solar_err) ** 2 +
                            (dR_v_max * self.v_max_err) ** 2 +
                            (dR_delta_v * self.delta_v_err) ** 2 +
                            (dR_R_solar * self.R_solar_err) ** 2)
            
    
            return R, R_err
        
    @staticmethod
    def exoplanet_prob_in_HZ(s_radius, teff):
        lum_sun = 3.83e26
        sigma = 5.67e-8
        
        lum = 4 * np.pi * ((s_radius * 1000) ** 2) * sigma * (teff ** 4)  # Stefan-Boltzmann Law
        hz_inner = (0.49 - (2.7619e-5 + 3.8095e-9) * (teff - 5700) ** 2) * (lum / lum_sun) ** 0.5  # AU
        hz_outer = (2.4 - (1.3786e-4 + 1.4286e-9) * (teff - 5700) ** 2) * (lum / lum_sun) ** 0.5  # AU
        
        w = hz_outer - hz_inner
        unnorm_av = 0.000343
        occurrence = w / unnorm_av
        rate = ((occurrence / 100) * 6.1) / 1096  # 6.1% of test planets are Terrans
        
        return max(rate, 0)  # Ensure non-negative values
    
    def __init__(self, v_max, delta_v, bv, feh):
        self.v_max = v_max
        self.delta_v = delta_v
        self.bv = bv
        self.feh = feh
        self.v_max_err = 50  # Example error
        self.delta_v_err = 0.2  # Example error
    
    def compute_exoplanet_probabilities(self):
        total = 0  # Initialize sum
        unc_total = 0
        zero_count = 0
        
        if isinstance(self.v_max, int):

            stars = self.AsteroseismologyScaling(self.v_max, self.delta_v, self.bv, self.feh, self.v_max_err, self.delta_v_err)
            rad, rad_err = stars.compute_radius()
            s_teff = stars.T_eff
            result = self.exoplanet_prob_in_HZ(rad, s_teff)
            unc = self.exoplanet_prob_in_HZ(rad_err, s_teff)

            print(f"Probability of exoplanet(s) in Habitable Zone for this star: {result:.4f} ± {unc:.4f}")
      
        else:
            for a in range(len(self.v_max)):
                stars = self.AsteroseismologyScaling(self.v_max[a], self.delta_v[a], self.bv[a], self.feh[a], self.v_max_err, self.delta_v_err)
                rad, rad_err = stars.compute_radius()
                s_teff = stars.T_eff
                result = self.exoplanet_prob_in_HZ(rad, s_teff)
                unc = self.exoplanet_prob_in_HZ(rad_err, s_teff)
                total += result
                unc_total += unc
                if result == 0:
                    zero_count += 1
        
            print(f"Total expected number of exoplanets in Habitable Zone from {len(self.v_max)} stars: {total:.4f} ± {unc_total:.4f}")
            print(f"Planets contributing zero probability: {zero_count}")
        return total, unc_total, zero_count


    def compute_transit_depth(self):
        stars = self.AsteroseismologyScaling(self.v_max, self.delta_v, self.bv, self.feh, self.v_max_err, self.delta_v_err)
        rad, rad_err = stars.compute_radius()
        self.R = rad
        self.R_err = rad_err
        self.rp_max = 2 * 6378.1  # Maximum planet radius in meters
        self.rp_min = 0.5 * 6378.1  # Minimum planet radius in meters
        self.rp_max_err = 2 * 0.001  # Maximum planet radius error
        self.rp_min_err = 0.5 * 0.001  # Minimum planet radius error
        
        D_max = (self.rp_max / self.R) ** 2
        D_min = (self.rp_min / self.R) ** 2
        
        D_max_err = np.sqrt(((2 * self.rp_max * self.rp_max_err) / self.R) ** 2 + 
                             ((-2 * self.R_err * self.rp_max ** 2) / self.R ** 3) ** 2)
        D_min_err = np.sqrt(((2 * self.rp_min * self.rp_min_err) / self.R) ** 2 + 
                             ((-2 * self.R_err * self.rp_min ** 2) / self.R ** 3) ** 2)
        
        print(f"Maximum depth: {D_max:.4e} ± {D_max_err:.4e}")
        print(f"Minimum depth: {D_min:.4e} ± {D_min_err:.4e}")
        
        return D_max, D_max_err, D_min, D_min_err
    
    def compute_orbital_period(self): # radius, radius_err, mass, mass_err
        """Computes habitable zone limits and orbital periods."""
        # Constants
        lum_sun = 3.83e26
        sigma = 5.67e-8
        G = 6.6743e-11
        au_to_m = 1.496e+11
        sec_per_day = 86400
        lum_sun_err = lum_sun * 0.03

        stars = self.AsteroseismologyScaling(self.v_max, self.delta_v, self.bv, self.feh, self.v_max_err, self.delta_v_err)
        rad, rad_err = stars.compute_radius()
        teff = stars.T_eff
        mass, mass_err = stars.compute_mass()
        
        # Compute stellar luminosity
        lum = 4 * np.pi * ((rad * 1000) ** 2) * sigma * (teff ** 4)
        lum_err = 4 * np.pi * ((rad_err * 1000) ** 2) * sigma * (teff ** 4)

        # Compute habitable zone boundaries (AU)
        hz_inner = (0.49 - (2.7619e-5 + 3.8095e-9) * (teff - 5700) ** 2) * (lum / lum_sun) ** 0.5
        hz_outer = (2.4 - (1.3786e-4 + 1.4286e-9) * (teff - 5700) ** 2) * (lum / lum_sun) ** 0.5
        hz_inner_err = (0.49 - (2.7619e-5 + 3.8095e-9) * (teff - 5700) ** 2) * (lum_err / lum_sun_err) ** 0.5
        hz_outer_err = (2.4 - (1.3786e-4 + 1.4286e-9) * (teff - 5700) ** 2) * (lum_err / lum_sun_err) ** 0.5

        # Convert AU to meters
        rad_inner, rad_outer = hz_inner * au_to_m, hz_outer * au_to_m
        rad_inner_err, rad_outer_err = hz_inner_err * au_to_m, hz_outer_err * au_to_m

        # Compute orbital periods using Kepler's Third Law
        period_min = np.sqrt((4 * np.pi ** 2 * rad_inner ** 3) / (G * mass)) / sec_per_day
        period_max = np.sqrt((4 * np.pi ** 2 * rad_outer ** 3) / (G * mass)) / sec_per_day
        period_min_err = np.sqrt((4 * np.pi ** 2 * rad_inner_err ** 3) / (G * mass_err)) / sec_per_day
        period_max_err = np.sqrt((4 * np.pi ** 2 * rad_outer_err ** 3) / (G * mass_err)) / sec_per_day

        T_dur_min = period_min * (rad/rad_outer)
        T_dur_max = period_max * (rad/rad_inner)

        print(f"Minimum orbital period: {period_min:.3f} ± {period_min_err:.3f} days")
        print(f"Maximum orbital period: {period_max:.3f} ± {period_max_err:.3f} days")
        
        return period_min, period_max, period_min_err, period_max_err, T_dur_min, T_dur_max

    def compute_prob_of_transit(self):
        #Constants
        lum_sun = 3.83e26
        sigma = 5.67e-8
        au_to_m = 1.496e+11

        
        stars = self.AsteroseismologyScaling(self.v_max, self.delta_v, self.bv, self.feh, self.v_max_err, self.delta_v_err)
        rad, rad_err = stars.compute_radius()
        teff = stars.T_eff


        # Compute stellar luminosity
        lum = 4 * np.pi * ((rad * 1000) ** 2) * sigma * (teff ** 4)
        lum_err = 4 * np.pi * ((rad_err * 1000) ** 2) * sigma * (teff ** 4)

        # Compute habitable zone boundaries (AU)
        hz_inner = (0.49 - (2.7619e-5 + 3.8095e-9) * (teff - 5700) ** 2) * (lum / lum_sun) ** 0.5
        hz_outer = (2.4 - (1.3786e-4 + 1.4286e-9) * (teff - 5700) ** 2) * (lum / lum_sun) ** 0.5

        # Convert AU to meters
        a_inner, a_outer = hz_inner * au_to_m, hz_outer * au_to_m
        min_prob = rad / a_outer
        max_prob = rad / a_inner

        print(f"Minimum probability of observation of transit: {min_prob:.7f}")
        print(f"Maximum probability of observation of transit: {max_prob:.7f}")

        return min_prob, max_prob


    @staticmethod
    def generate_transit_curve(period, depth, duration, time_resolution=500000):
        """Generates a synthetic transit light curve with a given duration."""
        time = np.linspace(0, period, time_resolution)
        flux = np.ones_like(time)
        mid_transit = period / 2  # Transit occurs at the midpoint
        transit_start = mid_transit - duration / 2
        transit_end = mid_transit + duration / 2
    
        transit_indices = (time >= transit_start) & (time <= transit_end)
        flux[transit_indices] -= depth  # Introduce the transit dip
    
        return time, flux, transit_start, transit_end

    def plot_transit_graphs(self, period_max, period_min, depth_max, depth_min, T_dur_max, T_dur_min, depth_max_err, depth_min_err):
        """Plots two transit observation graphs with separate zoomed-in transit regions."""
    
        # Generate transit curves
        time_max_1, flux_max_1, transit_start_max, transit_end_max = self.generate_transit_curve(period_max, depth_max, T_dur_max)
        time_max_2, flux_max_2, _, _ = self.generate_transit_curve(period_max, depth_min, T_dur_max)
        time_min_1, flux_min_1, transit_start_min, transit_end_min = self.generate_transit_curve(period_min, depth_max, T_dur_min)
        time_min_2, flux_min_2, _, _ = self.generate_transit_curve(period_min, depth_min, T_dur_min)
    
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)  # Different x-axis zoom for each
    
        # Calculate x-axis limits based on transit duration (1.2x buffer around transit)
        x_range_max = 1.2 * T_dur_max
        x_range_min = 1.2 * T_dur_min

        x_min_max = transit_start_max - x_range_max
        x_max_max = transit_end_max + x_range_max
        x_min_min = transit_start_min - x_range_min
        x_max_min = transit_end_min + x_range_min

        # Adjust y-axis limits based on transit depth
        y_min_max = 1 - 1.2 * depth_max  # Lower limit lower than 0 by 1.2*depth_max
        y_max_max = 1 + 0.1 * depth_max  # Upper limit higher than 1 by 0.1*depth_max

        y_min_min = 1 - 1.2 * depth_max  # Lower limit lower than 0 by 1.2*depth_min
        y_max_min = 1 + 0.1 * depth_max  # Upper limit higher than 1 by 0.1*depth_min

        # Plot max period transits
        axes[0].plot(time_max_1, flux_max_1, label=f'Max Period, Max Depth ({depth_max*100:.6f}%)', color='b')
        axes[0].plot(time_max_2, flux_max_2, label=f'Max Period, Min Depth ({depth_min*100:.6f}%)', color='r')
    
        # Add filled area around D_max and D_min with the corresponding errors
        axes[0].fill_between(time_max_1, 1 - (depth_max + depth_max_err), 1 - (depth_max - depth_max_err), 
                         color='blue', alpha=0.2, label=f'Max Depth Error ±{depth_max_err*100:.6f}%')
        axes[0].fill_between(time_max_2, 1 - (depth_min + depth_min_err), 1 - (depth_min - depth_min_err), 
                         color='red', alpha=0.2, label=f'Min Depth Error ±{depth_min_err*100:.6f}%')

        axes[0].set_xlim(x_min_max, x_max_max)
        axes[0].set_ylim(y_min_max, y_max_max)  # Adjust y-axis for max period plot
        axes[0].set_xlabel("Time (days)")
        axes[0].set_ylabel("Flux")
        axes[0].set_title("Zoomed Transits for Maximum Period")
        axes[0].legend()

        # Plot min period transits
        axes[1].plot(time_min_1, flux_min_1, label=f'Min Period, Max Depth ({depth_max*100:.6f}%)', color='b')
        axes[1].plot(time_min_2, flux_min_2, label=f'Min Period, Min Depth ({depth_min*100:.6f}%)', color='r')
    
        # Add filled area around D_max and D_min with the corresponding errors
        axes[1].fill_between(time_min_1, 1 - (depth_max + depth_max_err), 1 - (depth_max - depth_max_err), 
                         color='blue', alpha=0.2)
        axes[1].fill_between(time_min_2, 1 - (depth_min + depth_min_err), 1 - (depth_min - depth_min_err), 
                         color='red', alpha=0.2)

        axes[1].set_xlim(x_min_min, x_max_min)
        axes[1].set_ylim(y_min_min, y_max_min)  # Adjust y-axis for min period plot
        axes[1].set_xlabel("Time (days)")
        axes[1].set_ylabel("Flux")
        axes[1].set_title("Zoomed Transits for Minimum Period")
        axes[1].legend()
    
        plt.tight_layout()
        plt.show()


    def calc_mass_and_rad(self):
        stars = self.AsteroseismologyScaling(self.v_max, self.delta_v, self.bv, self.feh, self.v_max_err, self.delta_v_err)
        M, M_err = stars.compute_mass()
        R, R_err = stars.compute_radius()
        mass, mass_err = M / 1.988416e30, M_err / 1.988416e30
        radius, radius_err = R / 696340, R_err / 696340
        print(f"Mass of star: {mass:.7f} ± {mass_err:.7f} solar masses")
        print(f"Radius of star: {radius:.7f} ± {radius_err:.7f} solar radii")
        return mass, mass_err, radius, radius_err

        