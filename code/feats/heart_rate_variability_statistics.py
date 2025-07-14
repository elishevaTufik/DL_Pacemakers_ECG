#### By: Sebastian D. Goodfellow, Ph.D. (PhysioNet Challenge 2017)
#### Modified: Jonathan Rubin

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
import scipy as sp
from scipy import signal
from scipy import interpolate
import feats.pyeeg as pyeeg
from pyentrp import entropy as ent
from biosppy.signals.tools import smoother
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Local imports
from feats.utils.tools import *
from feats.utils.pyrem_univariate import *


class HeartRateVariabilityStatistics:

    """
    Generate a dictionary of heart rate variability statistics for one ECG signal.

    Parameters
    ----------
    ts : numpy array
        Full waveform time array.
    signal_raw : numpy array
        Raw full waveform.
    signal_filtered : numpy array
        Filtered full waveform.
    rpeaks : numpy array
        Array indices of R-Peaks
    templates_ts : numpy array
        Template waveform time array
    templates : numpy array
        Template waveforms
    fs : int, float
        Sampling frequency (Hz).
    template_before : float, seconds
            Time before R-Peak to start template.
    template_after : float, seconds
        Time after R-Peak to end template.

    Returns
    -------
    heart_rate_variability_statistics : dictionary
        Heart rate variability statistics.
    """

    def __init__(self, ts, signal_raw, signal_filtered, rpeaks,
                 templates_ts, templates, fs, template_before, template_after):

        # Input variables
        self.ts = ts
        self.signal_raw = signal_raw
        self.signal_filtered = signal_filtered
        self.rpeaks = rpeaks
        self.templates_ts = templates_ts
        self.templates = templates
        self.fs = fs
        self.template_before_ts = template_before
        self.template_after_ts = template_after
        self.template_before_sp = int(self.template_before_ts * self.fs)
        self.template_after_sp = int(self.template_after_ts * self.fs)

        # Set future variables
        self.heart_rate_ts = None
        self.heart_rate = None
        self.rri = None
        self.rri_ts = None
        self.diff_rri = None
        self.diff_rri_ts = None
        self.diff2_rri = None
        self.diff2_rri_ts = None
        self.templates_good = None
        self.templates_bad = None
        self.median_template = None
        self.median_template_good = None
        self.median_template_bad = None
        self.rpeaks_good = None
        self.rpeaks_bad = None
        self.templates_secondary = None
        self.median_template_secondary = None
        self.rpeaks_secondary = None

        # Calculate median template
        self.median_template = np.median(self.templates, axis=1)

        # R-Peak calculations
        self.template_rpeak_sp = self.template_before_sp

        # Correct R-Peak picks
        self.r_peak_check(correlation_threshold=0.9)

        # RR interval calculations
        self.rpeaks_ts = self.ts[self.rpeaks]
        self.calculate_rr_intervals(correlation_threshold=0.9)

        # Get secondary templates
        # self.get_secondary_templates(correlation_threshold=0.9)

        # Heart rate calculations
        self.calculate_heart_rate(smooth=True, size=3)

        # Feature dictionary
        self.heart_rate_variability_statistics = dict()

        # fs_spline = 10
        # diff_rri_ts_interp = np.arange(self.rri_ts[0], self.rri_ts[-1], 1 / float(fs_spline))
        # diff_tck = interpolate.splrep(self.rri_ts, self.rri, s=0)
        # diff_rri_interp = interpolate.splev(diff_rri_ts_interp, diff_tck, der=0)
        #
        # fig = plt.figure(figsize=(20, 5))
        # fig.subplots_adjust(wspace=0.6)
        #
        # ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        # ax3 = plt.subplot2grid((1, 3), (0, 2))
        #
        # ax1.plot(diff_rri_ts_interp, diff_rri_interp * 1000, '-', c=[0.7, 0.7, 0.7])
        # ax1.plot(self.rri_ts, self.rri * 1000, 'ok')
        #
        # ax2 = ax1.twinx()
        #
        # ax1.set_xlabel('Time, s', fontsize=30)
        # ax1.set_ylabel('rri, ms', fontsize=30)
        #
        # ax1.set_xlim([0, self.rri_ts.max()])
        # ax1.set_ylim([np.min(diff_rri_interp * 1000)-20, np.max(diff_rri_interp * 1000)+20])
        #
        # ax2.set_xlim([0, self.rri_ts.max()])
        # ax2.set_ylim([60 / (np.min(diff_rri_interp)-0.02), 60 / (np.max(diff_rri_interp)+0.02)])
        #
        # ax2.set_ylabel('Heart Rate, bpm', fontsize=30)
        #
        # ax1.tick_params(labelsize=20)
        # ax2.tick_params(labelsize=20)
        #
        # ax3.hist(self.rri*1000, bins=30, color=[0.7, 0.7, 0.7], edgecolor=[0.3, 0.3, 0.3], lw=0.5)
        #
        # ax3.tick_params(labelsize=20)
        # ax3.set_xlabel('rri, ms', fontsize=30)
        # ax3.set_ylabel('Count', fontsize=30)
        #
        # plt.show()
        #
        # print('Percentage of Unique Values: ' + str(len(np.unique(self.rri)) / len(self.rri) * 100))

    """
    Compile Features
    """
    def get_heart_rate_variability_statistics(self):
        return self.heart_rate_variability_statistics

    def calculate_heart_rate_variability_statistics(self):

        """
        Group features
        """
        self.heart_rate_variability_statistics.update(self.calculate_heart_rate_statistics(self.heart_rate))
        self.heart_rate_variability_statistics.update(
            self.calculate_rri_temporal_statistics(self.rri, self.diff_rri, self.diff2_rri)
        )
        self.heart_rate_variability_statistics.update(
            self.calculate_rri_nonlinear_statistics(self.rri, self.diff_rri, self.diff2_rri)
        )
        self.heart_rate_variability_statistics.update(
            self.calculate_pearson_correlation_statistics(self.rri)
        )
        self.heart_rate_variability_statistics.update(
            self.calculate_spearmanr_correlation_statistics(self.rri)
        )
        self.heart_rate_variability_statistics.update(
            self.calculate_kendalltau_correlation_statistics(self.rri)
        )
        self.heart_rate_variability_statistics.update(
            self.calculate_pointbiserialr_correlation_statistics(self.rri)
        )
        self.heart_rate_variability_statistics.update(
            self.calculate_poincare_statistics(self.rri)
        )
        self.heart_rate_variability_statistics.update(
            self.calculate_rri_spectral_statistics(self.rri, self.rri_ts)
        )
        self.heart_rate_variability_statistics.update(
            self.calculate_rri_fragmentation_statistics(self.diff_rri, self.diff_rri_ts)
        )
        self.heart_rate_variability_statistics.update(self.calculate_rpeak_detection_statistics())
        self.heart_rate_variability_statistics.update(self.calculate_rri_cluster_statistics())

    """
    Pre Processing
    """
    @staticmethod
    def normalize_series(series, method='median'):

        if method == 'median':
            return series / np.median(series)
        if method == 'mean':
            return series / np.mean(series)

    @staticmethod
    def is_outlier(points, thresh=8.0):

        if len(points.shape) == 1:
            points = points[:, None]

        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    def rri_physiological_filter(self):

        # Define physiologically possible interbeat interval range
        rri_max = 3.0   # 20 bpm
        rri_min = 0.25  # 240 bpm

        # get indices of physiologically impossible values
        possible = np.nonzero((self.rri <= rri_max) & (self.rri >= rri_min))[0]

    def get_secondary_templates(self, correlation_threshold=0.9):

        # Set rpeaks
        rpeaks = self.rpeaks_bad.astype(float)

        # If bad templates exist
        if self.templates_bad is not None:

            # Calculate median template
            self.median_template_secondary = np.median(self.templates_bad, axis=1)

            # Set counter
            count = 0

            # Loop through bad templates
            for template_id in range(self.templates_bad.shape[1]):

                # Calculate correlation coefficient
                correlation_coefficient = np.corrcoef(
                    self.median_template_secondary[self.template_rpeak_sp - 50:self.template_rpeak_sp + 50],
                    self.templates_bad[self.template_rpeak_sp - 50:self.template_rpeak_sp + 50, template_id]
                )

                # Check correlation
                if correlation_coefficient[0, 1] < correlation_threshold:

                    # Remove rpeak
                    rpeaks[template_id] = np.nan

                else:

                    # Update counter
                    count += 1

            if count >= 2:

                # Get good and bad rpeaks
                self.rpeaks_secondary = self.rpeaks_bad[np.isfinite(rpeaks)]

                # Get good and bad
                self.templates_secondary = self.templates_bad[:, np.where(np.isfinite(rpeaks))[0]]

                # Get median templates
                self.median_template_secondary = np.median(self.templates_secondary, axis=1)

    def r_peak_check(self, correlation_threshold=0.9):

        # Check lengths
        assert len(self.rpeaks) == self.templates.shape[1]

        # Loop through rpeaks
        for template_id in range(self.templates.shape[1]):

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, template_id]
            )

            # Check correlation
            if correlation_coefficient[0, 1] < correlation_threshold:

                # Compute cross correlation
                cross_correlation = signal.correlate(
                    self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                    self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, template_id]
                )

                # Correct rpeak
                rpeak_corrected = \
                    self.rpeaks[template_id] - \
                    (np.argmax(cross_correlation) -
                     len(self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25]))

                # Check to see if shifting the R-Peak improved the correlation coefficient
                if self.check_improvement(rpeak_corrected, correlation_threshold):

                    # Update rpeaks array
                    self.rpeaks[template_id] = rpeak_corrected

        # Re-extract templates
        self.templates, self.rpeaks = self.extract_templates(self.rpeaks)

        # Re-compute median template
        self.median_template = np.median(self.templates, axis=1)

        # Check lengths
        assert len(self.rpeaks) == self.templates.shape[1]

    def extract_templates(self, rpeaks):

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(self.signal_filtered)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - self.template_before_sp
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + self.template_after_sp
            if b > length:
                break

            # Append template list
            templates.append(self.signal_filtered[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    def check_improvement(self, rpeak_corrected, correlation_threshold):

        # Before R-Peak
        a = rpeak_corrected - self.template_before_sp

        # After R-Peak
        b = rpeak_corrected + self.template_after_sp

        if a >= 0 and b < len(self.signal_filtered):

            # Update template
            template_corrected = self.signal_filtered[a:b]

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                template_corrected[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25]
            )

            # Check new correlation
            if correlation_coefficient[0, 1] >= correlation_threshold:
                return True
            else:
                return False
        else:
            return False

    def calculate_rr_intervals(self, correlation_threshold=0.9):

        # Get rpeaks is floats
        rpeaks = self.rpeaks.astype(float)

        # Loop through templates
        for template_id in range(self.templates.shape[1]):

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, template_id]
            )

            # Check correlation
            if correlation_coefficient[0, 1] < correlation_threshold:

                # Remove rpeak
                rpeaks[template_id] = np.nan

        # RRI
        rri = np.diff(rpeaks) * 1 / self.fs
        rri_ts = rpeaks[0:-1] / self.fs + rri / 2

        # RRI Velocity
        diff_rri = np.diff(rri)
        diff_rri_ts = rri_ts[0:-1] + diff_rri / 2

        # RRI Acceleration
        diff2_rri = np.diff(diff_rri)
        diff2_rri_ts = diff_rri_ts[0:-1] + diff2_rri / 2

        # Drop rri, diff_rri, diff2_rri outliers
        self.rri = rri[np.isfinite(rri)]
        self.rri_ts = rri_ts[np.isfinite(rri_ts)]
        self.diff_rri = diff_rri[np.isfinite(diff_rri)]
        self.diff_rri_ts = diff_rri_ts[np.isfinite(diff_rri_ts)]
        self.diff2_rri = diff2_rri[np.isfinite(diff2_rri)]
        self.diff2_rri_ts = diff2_rri_ts[np.isfinite(diff2_rri_ts)]

        # Get good and bad rpeaks
        self.rpeaks_good = self.rpeaks[np.isfinite(rpeaks)]
        self.rpeaks_bad = self.rpeaks[~np.isfinite(rpeaks)]

        # Get good and bad
        self.templates_good = self.templates[:, np.where(np.isfinite(rpeaks))[0]]
        if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
            self.templates_bad = self.templates[:, np.where(~np.isfinite(rpeaks))[0]]

        # Get median templates
        self.median_template_good = np.median(self.templates_good, axis=1)
        if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
            self.median_template_bad = np.median(self.templates_bad, axis=1)

    def calculate_heart_rate(self, smooth=False, size=3):

        # compute heart rate
        self.heart_rate_ts = self.rri_ts
        self.heart_rate = 60. / self.rri

        # physiological limits
        index = np.nonzero(np.logical_and(self.heart_rate >= 20, self.heart_rate <= 240))
        self.heart_rate_ts = self.heart_rate_ts[index]
        self.heart_rate = self.heart_rate[index]

        # smooth with moving average
        if smooth and (len(self.heart_rate) > 1):
            self.heart_rate, _ = smoother(signal=self.heart_rate, kernel='boxcar', size=size, mirror=True)

    # TODO
    def extract_templates_rri(self, template_before_ts, template_after_ts):

        # Calculate before
        template_before_sp = self.template_rpeak_sp - int(template_before_ts * self.fs)
        template_after_sp = self.template_rpeak_sp + int(template_after_ts * self.fs)

        # Re-sample templates
        self.templates = self.templates[template_before_sp:template_after_sp, :]
        self.templates_good = self.templates[template_before_sp:template_after_sp, :]
        try:
            self.templates_bad = self.templates[template_before_sp:template_after_sp, :]
        except Exception:
            pass

        # Compute median templates
        self.median_template = np.median(self.templates, axis=1)
        self.median_template_good = np.median(self.templates, axis=1)
        try:
            self.median_template_bad = np.median(self.templates, axis=1)
        except Exception:
            pass

    """
    Feature Methods
    """

    @staticmethod
    def safe_check(value):

        try:
            if np.isfinite(value):
                return value
            else:
                return np.nan()

        except Exception:
            return np.nan

    def calculate_heart_rate_statistics(self, heart_rate, suffix=''):

        # Empty dictionary
        heart_rate_statistics = dict()

        # Calculate basic statistics
        if len(heart_rate) > 0:

            heart_rate_statistics['heart_rate_min'] = np.min(heart_rate)
            heart_rate_statistics['heart_rate_max'] = np.max(heart_rate)
            heart_rate_statistics['heart_rate_mean'] = np.mean(heart_rate)
            heart_rate_statistics['heart_rate_median'] = np.median(heart_rate)
            heart_rate_statistics['heart_rate_std'] = np.std(heart_rate, ddof=1)
            heart_rate_statistics['heart_rate_skew'] = sp.stats.skew(heart_rate)
            heart_rate_statistics['heart_rate_kurtosis'] = sp.stats.kurtosis(heart_rate)
        else:
            heart_rate_statistics['heart_rate_min'] = np.nan
            heart_rate_statistics['heart_rate_max'] = np.nan
            heart_rate_statistics['heart_rate_mean'] = np.nan
            heart_rate_statistics['heart_rate_median'] = np.nan
            heart_rate_statistics['heart_rate_std'] = np.nan
            heart_rate_statistics['heart_rate_skew'] = np.nan
            heart_rate_statistics['heart_rate_kurtosis'] = np.nan

        # Calculate non-linear statistics
        if len(heart_rate) > 1:
            heart_rate_statistics['heart_rate_approximate_entropy' + suffix] = \
                self.safe_check(pyeeg.ap_entropy(heart_rate, M=2, R=0.1*np.std(heart_rate)))
            heart_rate_statistics['heart_rate_sample_entropy' + suffix] = \
                self.safe_check(ent.sample_entropy(heart_rate, sample_length=2, tolerance=0.1*np.std(heart_rate))[0])
            heart_rate_statistics['heart_rate_multiscale_entropy' + suffix] = \
                self.safe_check(ent.multiscale_entropy(heart_rate, sample_length=2, tolerance=0.1*np.std(heart_rate))[0])
            heart_rate_statistics['heart_rate_permutation_entropy' + suffix] = \
                self.safe_check(ent.permutation_entropy(heart_rate, order=2, delay=1))
            heart_rate_statistics['heart_rate_multiscale_permutation_entropy' + suffix] = \
                self.safe_check(ent.multiscale_permutation_entropy(heart_rate, m=2, delay=1, scale=1)[0])
            heart_rate_statistics['heart_rate_fisher_info' + suffix] = fisher_info(heart_rate, tau=1, de=2)
            hjorth_parameters = hjorth(heart_rate)
            heart_rate_statistics['heart_rate_activity' + suffix] = hjorth_parameters[0]
            heart_rate_statistics['heart_rate_complexity' + suffix] = hjorth_parameters[1]
            heart_rate_statistics['heart_rate_morbidity' + suffix] = hjorth_parameters[2]
            heart_rate_statistics['heart_rate_hurst_exponent' + suffix] = pfd(heart_rate)
            heart_rate_statistics['heart_rate_svd_entropy' + suffix] = svd_entropy(heart_rate, tau=2, de=2)
            heart_rate_statistics['heart_rate_petrosian_fractal_dimension' + suffix] = pyeeg.pfd(heart_rate)
        else:
            heart_rate_statistics['heart_rate_approximate_entropy' + suffix] = np.nan
            heart_rate_statistics['heart_rate_sample_entropy' + suffix] = np.nan
            heart_rate_statistics['heart_rate_multiscale_entropy' + suffix] = np.nan
            heart_rate_statistics['heart_rate_permutation_entropy' + suffix] = np.nan
            heart_rate_statistics['heart_rate_multiscale_permutation_entropy' + suffix] = np.nan
            heart_rate_statistics['heart_rate_fisher_info' + suffix] = np.nan
            heart_rate_statistics['heart_rate_activity' + suffix] = np.nan
            heart_rate_statistics['heart_rate_complexity' + suffix] = np.nan
            heart_rate_statistics['heart_rate_morbidity' + suffix] = np.nan
            heart_rate_statistics['heart_rate_hurst_exponent' + suffix] = np.nan
            heart_rate_statistics['heart_rate_svd_entropy' + suffix] = np.nan
            heart_rate_statistics['heart_rate_petrosian_fractal_dimension' + suffix] = np.nan

        return heart_rate_statistics

    def calculate_rri_temporal_statistics(self, rri, diff_rri, diff2_rri, suffix=''):

        # Empty dictionary
        rri_temporal_statistics = dict()

        # RR interval statistics
        if len(rri) > 0:
            rri_temporal_statistics['rri_min' + suffix] = np.min(rri)
            rri_temporal_statistics['rri_max' + suffix] = np.max(rri)
            rri_temporal_statistics['rri_mean' + suffix] = np.mean(rri)
            rri_temporal_statistics['rri_median' + suffix] = np.median(rri)
            rri_temporal_statistics['rri_std' + suffix] = np.std(rri, ddof=1)
            rri_temporal_statistics['rri_skew' + suffix] = sp.stats.skew(rri)
            rri_temporal_statistics['rri_kurtosis' + suffix] = sp.stats.kurtosis(rri)
            rri_temporal_statistics['rri_rms' + suffix] = np.sqrt(np.mean(np.power(rri, 2)))
        else:
            rri_temporal_statistics['rri_min' + suffix] = np.nan
            rri_temporal_statistics['rri_max' + suffix] = np.nan
            rri_temporal_statistics['rri_mean' + suffix] = np.nan
            rri_temporal_statistics['rri_median' + suffix] = np.nan
            rri_temporal_statistics['rri_std' + suffix] = np.nan
            rri_temporal_statistics['rri_skew' + suffix] = np.nan
            rri_temporal_statistics['rri_kurtosis' + suffix] = np.nan
            rri_temporal_statistics['rri_rms' + suffix] = np.nan

        # Differences between successive RR interval differences statistics
        if len(diff_rri) > 0:
            rri_temporal_statistics['diff_rri_min' + suffix] = np.min(diff_rri)
            rri_temporal_statistics['diff_rri_max' + suffix] = np.max(diff_rri)
            rri_temporal_statistics['diff_rri_mean' + suffix] = np.mean(diff_rri)
            rri_temporal_statistics['diff_rri_median' + suffix] = np.median(diff_rri)
            rri_temporal_statistics['diff_rri_std' + suffix] = np.std(diff_rri, ddof=1)
            rri_temporal_statistics['diff_rri_skew' + suffix] = sp.stats.skew(diff_rri)
            rri_temporal_statistics['diff_rri_kurtosis' + suffix] = sp.stats.kurtosis(diff_rri)
            rri_temporal_statistics['diff_rri_rms' + suffix] = np.sqrt(np.mean(np.power(diff_rri, 2)))
        else:
            rri_temporal_statistics['diff_rri_min' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_max' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_mean' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_median' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_std' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_skew' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_kurtosis' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_rms' + suffix] = np.nan

        # Differences between successive RR intervals statistics
        if len(diff2_rri) > 0:
            rri_temporal_statistics['diff2_rri_min' + suffix] = np.min(diff2_rri)
            rri_temporal_statistics['diff2_rri_max' + suffix] = np.max(diff2_rri)
            rri_temporal_statistics['diff2_rri_mean' + suffix] = np.mean(diff2_rri)
            rri_temporal_statistics['diff2_rri_median' + suffix] = np.median(diff2_rri)
            rri_temporal_statistics['diff2_rri_std' + suffix] = np.std(diff2_rri, ddof=1)
            rri_temporal_statistics['diff2_rri_kurtosis' + suffix] = sp.stats.kurtosis(diff2_rri)
            rri_temporal_statistics['diff2_rri_rms' + suffix] = np.sqrt(np.mean(np.power(diff2_rri, 2)))
        else:
            rri_temporal_statistics['diff2_rri_min' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_max' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_mean' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_median' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_std' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_kurtosis' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_rms' + suffix] = np.nan

        # pNN statistics
        if len(diff_rri) > 0:
            rri_temporal_statistics['pnn01' + suffix] = self.pnn(diff_rri, 0.001)
            rri_temporal_statistics['pnn10' + suffix] = self.pnn(diff_rri, 0.01)
            rri_temporal_statistics['pnn20' + suffix] = self.pnn(diff_rri, 0.02)
            rri_temporal_statistics['pnn30' + suffix] = self.pnn(diff_rri, 0.03)
            rri_temporal_statistics['pnn40' + suffix] = self.pnn(diff_rri, 0.04)
            rri_temporal_statistics['pnn50' + suffix] = self.pnn(diff_rri, 0.05)
            rri_temporal_statistics['pnn60' + suffix] = self.pnn(diff_rri, 0.06)
            rri_temporal_statistics['pnn70' + suffix] = self.pnn(diff_rri, 0.07)
            rri_temporal_statistics['pnn80' + suffix] = self.pnn(diff_rri, 0.08)
            rri_temporal_statistics['pnn90' + suffix] = self.pnn(diff_rri, 0.09)
            rri_temporal_statistics['pnn100' + suffix] = self.pnn(diff_rri, 0.1)
            rri_temporal_statistics['pnn200' + suffix] = self.pnn(diff_rri, 0.2)
            rri_temporal_statistics['pnn400' + suffix] = self.pnn(diff_rri, 0.4)
            rri_temporal_statistics['pnn600' + suffix] = self.pnn(diff_rri, 0.6)
            rri_temporal_statistics['pnn800' + suffix] = self.pnn(diff_rri, 0.8)

        else:
            rri_temporal_statistics['pnn01' + suffix] = np.nan
            rri_temporal_statistics['pnn10' + suffix] = np.nan
            rri_temporal_statistics['pnn20' + suffix] = np.nan
            rri_temporal_statistics['pnn30' + suffix] = np.nan
            rri_temporal_statistics['pnn40' + suffix] = np.nan
            rri_temporal_statistics['pnn50' + suffix] = np.nan
            rri_temporal_statistics['pnn60' + suffix] = np.nan
            rri_temporal_statistics['pnn70' + suffix] = np.nan
            rri_temporal_statistics['pnn80' + suffix] = np.nan
            rri_temporal_statistics['pnn90' + suffix] = np.nan
            rri_temporal_statistics['pnn100' + suffix] = np.nan
            rri_temporal_statistics['pnn200' + suffix] = np.nan
            rri_temporal_statistics['pnn400' + suffix] = np.nan
            rri_temporal_statistics['pnn600' + suffix] = np.nan
            rri_temporal_statistics['pnn800' + suffix] = np.nan

        return rri_temporal_statistics

    @staticmethod
    def consecutive_count(random_list):

        retlist = []
        count = 1
        # Avoid IndexError for  random_list[i+1]
        for i in range(len(random_list) - 1):
            # Check if the next number is consecutive
            if random_list[i] + 1 == random_list[i+1]:
                count += 1
            else:
                # If it is not append the count and restart counting
                retlist = np.append(retlist, count)
                count = 1
        # Since we stopped the loop one early append the last count
        retlist = np.append(retlist, count)

        return retlist

    def calculate_rri_nonlinear_statistics(self, rri, diff_rri, diff2_rri, suffix=''):

        # Empty dictionary
        rri_nonlinear_statistics = dict()

        # Non-linear RR statistics
        if len(rri) > 1:
            rri_nonlinear_statistics['rri_approximate_entropy' + suffix] = \
                self.safe_check(pyeeg.ap_entropy(rri, M=2, R=0.1*np.std(rri)))
            rri_nonlinear_statistics['rri_sample_entropy' + suffix] = \
                self.safe_check(ent.sample_entropy(rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['rri_multiscale_entropy' + suffix] = \
                self.safe_check(ent.multiscale_entropy(rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['rri_permutation_entropy' + suffix] = \
                self.safe_check(ent.permutation_entropy(rri, order=2, delay=1))
            rri_nonlinear_statistics['rri_multiscale_permutation_entropy' + suffix] = \
                self.safe_check(ent.multiscale_permutation_entropy(rri, m=2, delay=1, scale=1)[0])
            rri_nonlinear_statistics['rri_fisher_info' + suffix] = fisher_info(rri, tau=1, de=2)
            hjorth_parameters = hjorth(rri)
            rri_nonlinear_statistics['rri_activity' + suffix] = hjorth_parameters[0]
            rri_nonlinear_statistics['rri_complexity' + suffix] = hjorth_parameters[1]
            rri_nonlinear_statistics['rri_morbidity' + suffix] = hjorth_parameters[2]
            rri_nonlinear_statistics['rri_hurst_exponent' + suffix] = pfd(rri)
            rri_nonlinear_statistics['rri_svd_entropy' + suffix] = svd_entropy(rri, tau=2, de=2)
            rri_nonlinear_statistics['rri_petrosian_fractal_dimension' + suffix] = pyeeg.pfd(rri)
        else:
            rri_nonlinear_statistics['rri_approximate_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_sample_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_multiscale_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_multiscale_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_fisher_info' + suffix] = np.nan
            rri_nonlinear_statistics['rri_activity' + suffix] = np.nan
            rri_nonlinear_statistics['rri_complexity' + suffix] = np.nan
            rri_nonlinear_statistics['rri_morbidity' + suffix] = np.nan
            rri_nonlinear_statistics['rri_hurst_exponent' + suffix] = np.nan
            rri_nonlinear_statistics['rri_svd_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_petrosian_fractal_dimension' + suffix] = np.nan

        # Non-linear RR difference statistics
        if len(diff_rri) > 1:
            rri_nonlinear_statistics['diff_rri_approximate_entropy' + suffix] = \
                self.safe_check(pyeeg.ap_entropy(diff_rri, M=2, R=0.1*np.std(rri)))
            rri_nonlinear_statistics['diff_rri_sample_entropy' + suffix] = \
                self.safe_check(ent.sample_entropy(diff_rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['diff_rri_multiscale_entropy' + suffix] = \
                self.safe_check(ent.multiscale_entropy(diff_rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['diff_rri_permutation_entropy' + suffix] = \
                self.safe_check(ent.permutation_entropy(diff_rri, order=2, delay=1))
            rri_nonlinear_statistics['diff_rri_multiscale_permutation_entropy' + suffix] = \
                self.safe_check(ent.multiscale_permutation_entropy(diff_rri, m=2, delay=1, scale=1)[0])
            rri_nonlinear_statistics['diff_rri_fisher_info' + suffix] = fisher_info(diff_rri, tau=1, de=2)
            hjorth_parameters = hjorth(diff_rri)
            rri_nonlinear_statistics['diff_rri_activity' + suffix] = hjorth_parameters[0]
            rri_nonlinear_statistics['diff_rri_complexity' + suffix] = hjorth_parameters[1]
            rri_nonlinear_statistics['diff_rri_morbidity' + suffix] = hjorth_parameters[2]
            rri_nonlinear_statistics['diff_rri_hurst_exponent' + suffix] = pfd(diff_rri)
            rri_nonlinear_statistics['diff_rri_svd_entropy' + suffix] = svd_entropy(diff_rri, tau=2, de=2)
            rri_nonlinear_statistics['diff_rri_petrosian_fractal_dimension' + suffix] = pyeeg.pfd(diff_rri)
        else:
            rri_nonlinear_statistics['diff_rri_approximate_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_sample_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_multiscale_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_multiscale_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_fisher_info' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_activity' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_complexity' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_morbidity' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_hurst_exponent' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_svd_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_petrosian_fractal_dimension' + suffix] = np.nan

        # Non-linear RR difference difference statistics
        if len(diff2_rri) > 1:
            rri_nonlinear_statistics['diff2_rri_shannon_entropy' + suffix] = \
                self.safe_check(ent.shannon_entropy(diff2_rri))
            rri_nonlinear_statistics['diff2_rri_approximate_entropy' + suffix] = \
                self.safe_check(pyeeg.ap_entropy(diff2_rri, M=2, R=0.1*np.std(rri)))
            rri_nonlinear_statistics['diff2_rri_sample_entropy' + suffix] = \
                self.safe_check(ent.sample_entropy(diff2_rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['diff2_rri_multiscale_entropy' + suffix] = \
                self.safe_check(ent.multiscale_entropy(diff2_rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['diff2_rri_permutation_entropy' + suffix] = \
                self.safe_check(ent.permutation_entropy(diff2_rri, order=2, delay=1))
            rri_nonlinear_statistics['diff2_rri_multiscale_permutation_entropy' + suffix] = \
                self.safe_check(ent.multiscale_permutation_entropy(diff2_rri, m=2, delay=1, scale=1)[0])
            rri_nonlinear_statistics['diff2_rri_fisher_info' + suffix] = fisher_info(diff2_rri, tau=1, de=2)
            hjorth_parameters = hjorth(diff2_rri)
            rri_nonlinear_statistics['diff2_rri_activity' + suffix] = hjorth_parameters[0]
            rri_nonlinear_statistics['diff2_rri_complexity' + suffix] = hjorth_parameters[1]
            rri_nonlinear_statistics['diff2_rri_morbidity' + suffix] = hjorth_parameters[2]
            rri_nonlinear_statistics['diff2_rri_hurst_exponent' + suffix] = pfd(diff2_rri)
            rri_nonlinear_statistics['diff2_rri_svd_entropy' + suffix] = svd_entropy(diff2_rri, tau=2, de=2)
            rri_nonlinear_statistics['diff2_rri_petrosian_fractal_dimension' + suffix] = pyeeg.pfd(diff2_rri)
        else:
            rri_nonlinear_statistics['diff2_rri_shannon_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_approximate_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_sample_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_multiscale_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_multiscale_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_fisher_info' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_activity' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_complexity' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_morbidity' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_hurst_exponent' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_svd_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_petrosian_fractal_dimension' + suffix] = np.nan

        return rri_nonlinear_statistics

    @staticmethod
    def pnn(diff_rri, time):

        # Count number of rri diffs greater than the specified time
        nn = sum(abs(diff_rri) > time)

        # Compute pNN
        pnn = nn / len(diff_rri) * 100

        return pnn

    @staticmethod
    def calculate_pearson_correlation_statistics(rri, suffix=''):
      
        # Empty dictionary
        pearson_correlation_statistics = dict()

        # Calculate Pearson correlation
        if len(rri[0:-2]) >= 2 and len(rri[1:-1]) >= 2:
            pearson_coeff_p1, pearson_p_value_p1 = sp.stats.pearsonr(rri[0:-2], rri[1:-1])
        else:
            pearson_coeff_p1, pearson_p_value_p1 = np.nan, np.nan
            
        if len(rri[0:-3]) >= 2 and len(rri[2:-1]) >= 2:
            pearson_coeff_p2, pearson_p_value_p2 = sp.stats.pearsonr(rri[0:-3], rri[2:-1])
        else:
            pearson_coeff_p2, pearson_p_value_p2 = np.nan, np.nan
            
        if len(rri[0:-4]) >= 2 and len(rri[3:-1]) >= 2:
            pearson_coeff_p3, pearson_p_value_p3 = sp.stats.pearsonr(rri[0:-4], rri[3:-1])
        else:
            pearson_coeff_p3, pearson_p_value_p3 = np.nan, np.nan

        # Get features
        pearson_correlation_statistics['rri_p1_pearson_coeff' + suffix] = pearson_coeff_p1
        pearson_correlation_statistics['rri_p1_pearson_p_value' + suffix] = pearson_p_value_p1
        pearson_correlation_statistics['rri_p2_pearson_coeff' + suffix] = pearson_coeff_p2
        pearson_correlation_statistics['rri_p2_pearson_p_value' + suffix] = pearson_p_value_p2
        pearson_correlation_statistics['rri_p3_pearson_coeff' + suffix] = pearson_coeff_p3
        pearson_correlation_statistics['rri_p3_pearson_p_value' + suffix] = pearson_p_value_p3

        return pearson_correlation_statistics

    @staticmethod
    def calculate_spearmanr_correlation_statistics(rri, suffix=''):

        # Empty dictionary
        spearmanr_correlation_statistics = dict()

        # Calculate Pearson correlation
        if len(rri[0:-2]) >= 2 and len(rri[1:-1]) >= 2:
            spearmanr_coeff_p1, spearmanr_p_value_p1 = sp.stats.spearmanr(rri[0:-2], rri[1:-1])
        else:
            spearmanr_coeff_p1, spearmanr_p_value_p1 = np.nan, np.nan

        # Get features
        spearmanr_correlation_statistics['rri_p1_spearmanr_coeff' + suffix] = spearmanr_coeff_p1
        spearmanr_correlation_statistics['rri_p1_spearmanr_p_value' + suffix] = spearmanr_p_value_p1

        return spearmanr_correlation_statistics

    @staticmethod
    def calculate_kendalltau_correlation_statistics(rri, suffix=''):

        # Empty dictionary
        kendalltau_correlation_statistics = dict()

        # Calculate Pearson correlation
        if len(rri[0:-2]) >= 2 and len(rri[1:-1]) >= 2:
            kendalltau_coeff_p1, kendalltau_p_value_p1 = sp.stats.kendalltau(rri[0:-2], rri[1:-1])
        else:
            kendalltau_coeff_p1, kendalltau_p_value_p1 = np.nan, np.nan

        # Get features
        kendalltau_correlation_statistics['rri_p1_kendalltau_coeff' + suffix] = kendalltau_coeff_p1
        kendalltau_correlation_statistics['rri_p1_kendalltau_p_value' + suffix] = kendalltau_p_value_p1

        return kendalltau_correlation_statistics

    @staticmethod
    def calculate_pointbiserialr_correlation_statistics(rri, suffix=''):

        # Empty dictionary
        pointbiserialr_correlation_statistics = dict()

        # Calculate Pearson correlation
        if len(rri[0:-2]) >= 2 and len(rri[1:-1]) >= 2:
            pointbiserialr_coeff_p1, pointbiserialr_p_value_p1 = sp.stats.pointbiserialr(rri[0:-2], rri[1:-1])
        else:
            pointbiserialr_coeff_p1, pointbiserialr_p_value_p1 = np.nan, np.nan

        # Get features
        pointbiserialr_correlation_statistics['rri_p1_pointbiserialr_coeff' + suffix] = pointbiserialr_coeff_p1
        pointbiserialr_correlation_statistics['rri_p1_pointbiserialr_p_value' + suffix] = pointbiserialr_p_value_p1

        return pointbiserialr_correlation_statistics

    def calculate_poincare_statistics(self, rri, suffix=''):

        # Empty dictionary
        poincare_statistics = dict()

        # Calculate poincare statistics
        sd1, sd2 = self.poincare(rri)

        # Get features
        poincare_statistics['poincare_sd1' + suffix] = sd1
        poincare_statistics['poincare_sd2' + suffix] = sd2

        return poincare_statistics

    @staticmethod
    def poincare(rri):

        # Calculate the difference between successive rri's
        diff_rri = np.diff(rri)

        # Calculate SD1 and SD2
        sd1 = np.sqrt(np.std(diff_rri, ddof=1) ** 2 * 0.5)
        sd2 = np.sqrt(2 * np.std(rri, ddof=1) ** 2 - 0.5 * np.std(diff_rri, ddof=1) ** 2)

        return sd1, sd2

    @staticmethod
    def calculate_rri_spectral_statistics(rri, rri_ts, suffix=''):

        # Empty dictionary
        rri_spectral_statistics = dict()

        if len(rri) > 3:

            # Zero the time array
            rri_ts = rri_ts - rri_ts[0]

            # Set resampling rate
            fs = 10  # Hz

            # Generate new resampling time array
            rri_ts_interp = np.arange(rri_ts[0], rri_ts[-1], 1 / float(fs))

            # Setup interpolation function
            tck = interpolate.splrep(rri_ts, rri, s=0)

            # Interpolate rri on new time array
            rri_interp = interpolate.splev(rri_ts_interp, tck, der=0)

            # Set frequency band limits [Hz]
            vlf_band = (0, 0.04)    # Very low frequency
            lf_band = (0.04, 0.15)  # Low frequency
            hf_band = (0.15, 0.6)   # High frequency
            vhf_band = (0.6, 2)   # High frequency

            # Compute Welch periodogram
            fxx, pxx = signal.welch(x=rri_interp, fs=fs)

            # Get frequency band indices
            vlf_index = np.logical_and(fxx >= vlf_band[0], fxx < vlf_band[1])
            lf_index = np.logical_and(fxx >= lf_band[0], fxx < lf_band[1])
            hf_index = np.logical_and(fxx >= hf_band[0], fxx < hf_band[1])
            vhf_index = np.logical_and(fxx >= vhf_band[0], fxx < vhf_band[1])

            # Compute power in each frequency band
            vlf_power = np.trapz(y=pxx[vlf_index], x=fxx[vlf_index])
            lf_power = np.trapz(y=pxx[lf_index], x=fxx[lf_index])
            hf_power = np.trapz(y=pxx[hf_index], x=fxx[hf_index])
            vhf_power = np.trapz(y=pxx[vhf_index], x=fxx[vhf_index])

            # Compute total power
            total_power = vlf_power + lf_power + hf_power + vhf_power

            # Compute spectral ratios
            rri_spectral_statistics['rri_low_high_spectral_ratio' + suffix] = lf_power / hf_power
            rri_spectral_statistics['rri_low_very_high_spectral_ratio' + suffix] = lf_power / vhf_power
            rri_spectral_statistics['rri_low_frequency_power' + suffix] = (lf_power / total_power) * 100
            rri_spectral_statistics['rri_high_frequency_power' + suffix] = (hf_power / total_power) * 100
            rri_spectral_statistics['rri_very_high_frequency_power' + suffix] = (vhf_power / total_power) * 100
            rri_spectral_statistics['rri_freq_max_frequency_power' + suffix] = \
                fxx[np.argmax(pxx[np.logical_and(fxx >= lf_band[0], fxx < vhf_band[1])])]
            rri_spectral_statistics['rri_power_max_frequency_power' + suffix] = \
                np.max(pxx[np.logical_and(fxx >= lf_band[0], fxx < vhf_band[1])])
            rri_spectral_statistics['rri_spectral_entropy' + suffix] = \
                spectral_entropy(rri_interp, sampling_freq=fs, bands=[0.04, 0.15, 0.4])

        else:
            # Compute spectral ratios
            rri_spectral_statistics['rri_low_high_spectral_ratio' + suffix] = np.nan
            rri_spectral_statistics['rri_low_very_high_spectral_ratio' + suffix] = np.nan
            rri_spectral_statistics['rri_low_frequency_power' + suffix] = np.nan
            rri_spectral_statistics['rri_high_frequency_power' + suffix] = np.nan
            rri_spectral_statistics['rri_very_high_frequency_power' + suffix] = np.nan
            rri_spectral_statistics['rri_freq_max_frequency_power' + suffix] = np.nan
            rri_spectral_statistics['rri_power_max_frequency_power' + suffix] = np.nan
            rri_spectral_statistics['rri_spectral_entropy' + suffix] = np.nan

        return rri_spectral_statistics

    @staticmethod
    def calculate_rri_fragmentation_statistics(diff_rri, diff_rri_ts, suffix=''):

        # Empty dictionary
        rri_fragmentation_statistics = dict()

        if len(diff_rri) > 1:
            # Calculate zero crossing indices.
            diff_rri_zero_crossings = np.where(np.diff(np.sign(diff_rri)))[0]

            # The percentage of zero-crossing points in the rri diff time series.
            rri_fragmentation_statistics['fragmentation_pip' + suffix] = \
                len(diff_rri_zero_crossings) / len(diff_rri) * 100

            # The inverse of the average length of the acceleration/deceleration segments.
            rri_fragmentation_statistics['fragmentation_ials' + suffix] = \
                np.mean(1 / np.diff(diff_rri_ts[diff_rri_zero_crossings]))

            # The percentage of NN intervals in acceleration and deceleration segments with three or more NN intervals.
            rri_fragmentation_statistics['fragmentation_pss' + suffix] = \
                np.sum(np.diff(diff_rri_zero_crossings) >= 3) / len(diff_rri_zero_crossings) * 100

            # The percentage of NN intervals in alternation segments.
            rri_fragmentation_statistics['fragmentation_pas' + suffix] = \
                np.sum(np.diff(diff_rri_zero_crossings) == 1) / len(diff_rri_zero_crossings) * 100

        else:
            rri_fragmentation_statistics['fragmentation_pip' + suffix] = np.nan
            rri_fragmentation_statistics['fragmentation_ials' + suffix] = np.nan
            rri_fragmentation_statistics['fragmentation_pss' + suffix] = np.nan
            rri_fragmentation_statistics['fragmentation_pas' + suffix] = np.nan

        return rri_fragmentation_statistics

    def calculate_rpeak_detection_statistics(self):

        # Empty dictionary
        rpeak_detection_statistics = dict()

        # Get median rri
        if len(self.rri) > 0:

            # Compute median rri
            rri_avg = np.median(self.rri)

        else:

            # Define possible rri's
            th1 = 1.5  # 40 bpm
            th2 = 0.3  # 200 bpm

            # Compute mean rri
            rri_avg = (th1 + th2) / 2

        # Calculate waveform duration in seconds
        time_duration = np.max(self.ts)

        # Calculate theoretical number of expected beats
        beat_count_theory = np.ceil(time_duration / rri_avg)

        # Calculate percentage of observed beats to theoretical beats
        rpeak_detection_statistics['detection_success'] = len(self.rpeaks) / beat_count_theory

        # Calculate percentage of bad rpeaks
        if self.rpeaks_bad is None:
            rpeak_detection_statistics['rpeaks_bad'] = 0.0
        else:
            rpeak_detection_statistics['rpeaks_bad'] = len(self.rpeaks_bad) / len(self.rpeaks)

        return rpeak_detection_statistics

    def calculate_rri_cluster_statistics(self):

        # Empty dictionary
        rri_cluster_statistics = dict()

        if len(self.rri) > 6:

            # Combine r_ibi and r_ibi + 1
            rri_combined = np.column_stack((self.rri[0:-2], self.rri[1:-1]))

            # Set cluster range
            clusters = range(1, 4)

            # Compute KMeans clusters
            cluster_models = [KMeans(n_clusters=cluster).fit(rri_combined) for cluster in clusters]

            # Get centroids
            centroids = [cluster_model.cluster_centers_ for cluster_model in cluster_models]

            # Compute intra-cluster distances
            distances = [cdist(rri_combined, cent, 'euclidean') for cent in centroids]
            dist = [np.min(distance, axis=1) for distance in distances]
            ssd = [sum(d) / rri_combined.shape[0] * 1000 for d in dist]

            rri_cluster_statistics['rri_cluster_ssd_slope'] = np.polyfit(x=clusters, y=ssd, deg=1)[0]
            rri_cluster_statistics['rri_cluster_ssd_1'] = ssd[0]
            rri_cluster_statistics['rri_cluster_ssd_2'] = ssd[1]
            rri_cluster_statistics['rri_cluster_ssd_3'] = ssd[2]

        else:
            rri_cluster_statistics['rri_cluster_ssd_slope'] = np.nan
            rri_cluster_statistics['rri_cluster_ssd_1'] = np.nan
            rri_cluster_statistics['rri_cluster_ssd_2'] = np.nan
            rri_cluster_statistics['rri_cluster_ssd_3'] = np.nan

        return rri_cluster_statistics
