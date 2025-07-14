#### By: Sebastian D. Goodfellow, Ph.D. (PhysioNet Challenge 2017)
#### Modified: Jonathan Rubin

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import scipy.io as sio
import pandas as pd
import os
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal

# Local imports
from feats.full_waveform_statistics import *
from feats.heart_rate_variability_statistics import *
from feats.template_statistics import *


class Features:

    """
    Generate an array of ECG features.

    Parameters
    ----------
    file_path : string
        Path for ECG .mat files.
    fs : int, float
        Sampling frequency (Hz).
    feature_groups : list
        List of feature groups to include when generating feature array.
    labels : DataFrame
        Training Labels.
    """

    available_feature_groups = [
        'full_waveform_statistics', 'heart_rate_variability_statistics', 'template_statistics'
    ]

    def __init__(self, data, fs, feature_groups, labels=None):

        # Input attributes
        self.data = data
        self.fs = fs
        self.labels = labels
        self.feature_groups = feature_groups
        self.features = None

    def get_features(self):
        return self.features

    @classmethod
    def get_feature_groups(cls):
        return cls.available_feature_groups

    def calculate_features(self, filter_bandwidth, show=False,
                           channel=0, normalize=True, polarity_check=True,
                           template_before=0.2, template_after=0.4):

        """
        Get ECG feature DataFrame

        Parameters
        ----------
        show : boolean
            Print processing progress.
        filter_bandwidth : list
            Bandpass filter limits [low, high]
        channel: int
            Channel number
        normalize : boolean
            Should the signal be normalized.
        polarity_check : boolean
            Should a polarity check be conducted.
        template_before : float, seconds
            Time before R-Peak to start template.
        template_after : float, seconds
            Time after R-Peak to end template.
        """

        # Create empty features DataFrame
        self.features = pd.DataFrame()

        # # Load ECG .mat file
        # try:

        # Load ECG signal
        signal_raw = self.data

        # Preprocess signal
        ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates = \
            self._preprocess_signal(signal_raw, filter_bandwidth, normalize,
                                    polarity_check, template_before, template_after)

        # Get features
        self.features = self.features._append(
            self._group_features(
                file_name='',
                ts=ts,
                signal_raw=signal_raw,
                signal_filtered=signal_filtered,
                rpeaks=rpeaks,
                templates_ts=templates_ts,
                templates=templates,
                template_before=template_before,
                template_after=template_after
            ),
            ignore_index=True
        )

    def _preprocess_signal(self, signal_raw, filter_bandwidth, normalize,
                           polarity_check, template_before, template_after):

        # Filter signal
        signal_filtered = self._apply_filter(signal_raw, filter_bandwidth)

        # Get BioSPPy ECG object
        ecg_object = ecg.ecg(signal=signal_raw, sampling_rate=self.fs, show=False)

        # Get BioSPPy output
        ts = ecg_object['ts']                        # Signal time array
        rpeaks = ecg_object['rpeaks']                # rpeak indices

        # Get templates and template time array
        templates, rpeaks = self._extract_templates(signal_filtered, rpeaks, template_before, template_after)
        templates_ts = np.linspace(-template_before, template_after, templates.shape[1], endpoint=False)

        # Polarity check
        if polarity_check:

            # Get extremes of median templates
            templates_min = np.min(np.median(templates, axis=1))
            templates_max = np.max(np.median(templates, axis=1))

            if np.abs(templates_min) > np.abs(templates_max):

                # Flip polarity
                signal_raw *= -1
                signal_filtered *= -1
                templates *= -1

        # Normalize waveform
        if normalize:

            # Get median templates max
            templates_max = np.max(np.median(templates, axis=1))

            # Normalize ECG signals
            signal_raw /= templates_max
            signal_filtered /= templates_max
            templates /= templates_max

        return ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates

    def _extract_templates(self, signal, rpeaks, before, after):

        # convert delimiters to samples
        before = int(before * self.fs)
        after = int(after * self.fs)

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(signal)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - before
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + after
            if b > length:
                break

            # Append template list
            templates.append(signal[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    def _apply_filter(self, signal, filter_bandwidth):

        # Calculate filter order
        order = int(0.3 * self.fs)

        # Filter signal
        signal, _, _ = filter_signal(signal=signal,
                                     ftype='FIR',
                                     band='bandpass',
                                     order=order,
                                     frequency=filter_bandwidth,
                                     sampling_rate=self.fs)

        return signal

    def _group_features(self, file_name, ts, signal_raw, signal_filtered, rpeaks,
                        templates_ts, templates, template_before, template_after):

        """
        Get a dictionary of all ECG features
        """

        # Empty features dictionary
        features = dict()

        # Set ECG file name
        features['file_name'] = file_name

        # Loop through feature groups
        for feature_group in self.feature_groups:

            ##########################
            # Full Waveform Statistics
            ##########################
            if feature_group == 'full_waveform_statistics':

                # Get features
                full_waveform_statistics = FullWaveformStatistics(
                    ts=ts,
                    signal_raw=signal_raw,
                    signal_filtered=signal_filtered,
                    rpeaks=rpeaks,
                    templates_ts=templates_ts,
                    templates=templates,
                    fs=self.fs,
                    template_before=template_before,
                    template_after=template_after
                )
                full_waveform_statistics.calculate_full_waveform_statistics()

                # Update feature dictionary
                features.update(full_waveform_statistics.get_full_waveform_statistics())

            ###################
            # Heart Rate Variability Statistics
            ###################
            if feature_group == 'heart_rate_variability_statistics':

                # Get features
                heart_rate_variability_statistics = HeartRateVariabilityStatistics(
                    ts=ts,
                    signal_raw=signal_raw,
                    signal_filtered=signal_filtered,
                    rpeaks=rpeaks,
                    templates_ts=templates_ts,
                    templates=templates,
                    fs=self.fs,
                    template_before=template_before,
                    template_after=template_after
                )
                heart_rate_variability_statistics.calculate_heart_rate_variability_statistics()

                # Update feature dictionary
                features.update(heart_rate_variability_statistics.get_heart_rate_variability_statistics())

            #####################
            # Template Statistics
            #####################
            if feature_group == 'template_statistics':

                # Get features
                template_statistics = TemplateStatistics(
                    ts=ts,
                    signal_raw=signal_raw,
                    signal_filtered=signal_filtered,
                    rpeaks=rpeaks,
                    templates_ts=templates_ts,
                    templates=templates,
                    fs=self.fs,
                    template_before=template_before,
                    template_after=template_after
                )
                template_statistics.calculate_template_statistics()

                # Update feature dictionary
                features.update(template_statistics.get_template_statistics())

        return pd.Series(data=features)
