"""
utility functions for multi-modal data handling.
extracts modality info from metadata and upsamples low rate signals.
"""

import numpy as np
import logging
logger = logging.getLogger(__name__)


def get_modality_info(metadata):
    if 'modality_info' in metadata:
        return metadata['modality_info']

    # for backwards compatibility: old .meta files only have volts_per_bit
    logger.debug("No modality_info found, defaulting to voltage")
    return {
        'modality_type': 'voltage',
        'unit': 'V',
        'scale_factor': metadata.get('volts_per_bit', 1.0)
    }


def upsample_data(data, ratio, method='zero_order_hold'):
    """
    upsample data by zero order hold or interpolating.

    zero-order hold repeats each sample `ratio` times. 
    - appropriate for sparse signals like EMG variance or temperature where interpolating 
    would create artificial values between measurements.

    Args:
        data: shape (n_samples,) or (n_samples, n_channels)
        ratio: upsampling factor (e.g. 800 for 0.25Hz -> 200Hz)
        method: 'zero_order_hold' or 'linear'

    E.g: upsample_data(np.array([1, 2, 3]), ratio=4)
        >>> array([1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3])
    """
    if ratio == 1:
        return data

    if method == 'zero_order_hold':
        # np.repeat copies each value `ratio` times along the time axis
        axis = 0 if data.ndim > 1 else None
        return np.repeat(data, ratio, axis=axis)

    elif method == 'linear':
        old_indices = np.arange(len(data)) * ratio
        new_indices = np.arange(len(data) * ratio)

        if data.ndim == 1:
            return np.interp(new_indices, old_indices, data)
        else:
            # interpolate each channel separately
            upsampled = np.zeros((len(data) * ratio, data.shape[1]))
            for ch in range(data.shape[1]):
                upsampled[:, ch] = np.interp(new_indices, old_indices, data[:, ch])
            return upsampled

    else:
        raise ValueError(f"Unknown method: {method}. Use 'zero_order_hold' or 'linear'")
