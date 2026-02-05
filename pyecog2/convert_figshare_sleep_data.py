"""
conversion script for FigShare sleep dataset to separate files for multi modal support.

Reads original files:
- .dat files: sleep scores, power spectra, EEG variance, EMG variance, temperature
- .eeg files: raw 200 Hz EEG data

Converts to PyEcog format with separate files per data type:
{x}_EEG.bin 200 Hz raw voltage
{x}_EMG.bin (sample every 4s), EMG variance
{x}_Temperature.bin (sample every 4s), temperature (if recorded)
{x}_SleepScore.bin (sample every 4s), categorical sleep states

Each .bin gets a .meta file with modality info
"""

import numpy as np
import os
import logging
from datetime import datetime
from pyecog2.ProjectClass import create_metafile_for_modality
logger = logging.getLogger(__name__)

def readbinary_dat(file):
    """
    reads .dat file from the FigShare sleep dataset.

    within figshare dataset each file has 86400 records (one per 4 second epoch, 96 hours total).
    Each rrecord is: 1 byte score + 401 float32 spectrum + 3 float32 (EEG var, EMG var, temp).
    """
    # defining dtype of each record
    # so numpy can read all 86400 records in one call instead of looping.
    record_dtype = np.dtype([
        ('score', 'S1'),           # 1 byte: sleep state char ('w', 'nrem', or 'rem')
        ('spectrum', 'f4', (401,)),  # 401 floats: power spectrum bins (0-100Hz at 0.25Hz)
        ('misc', 'f4', (3,))        # 3 floats: EEG variance, EMG variance, temperature
    ])
    records = np.fromfile(file, dtype=record_dtype)

    # Unpack fields from the structured array
    sleep_scores = records['score'].tobytes().decode('utf-8')
    power_spectra = records['spectrum']
    EEG_variance = records['misc'][:, 0].copy() # made a copy so it's a standalone array
    EMG_variance = records['misc'][:, 1].copy()
    Temperature = records['misc'][:, 2].copy()

    # assuming temp not recorded when 0 at the start
    if Temperature[0] == 0:
        Temperature = np.array([])

    logger.info(f"Read {file}: {len(sleep_scores)} epochs, temp recorded: {len(Temperature) > 0}")
    return sleep_scores, power_spectra, EEG_variance, EMG_variance, Temperature


def readbinary_EEG(file):
    """Read raw EEG binary file. 200 Hz float32, typically ~69M samples (96 hours)."""
    raw_EEG = np.fromfile(file, dtype=np.float32)
    logger.info(f"Read {file}: {len(raw_EEG)} samples ({len(raw_EEG)/200/3600:.1f} hours)")
    return raw_EEG


def convert_animal_to_multimodal(dat_file, eeg_file, output_folder, animal_id=None):
    """
    Convert one animal's data into separate per modality .bin + .meta files.

    Returns dict mapping modality to its .meta file path.
    """
    if animal_id is None:
        animal_id = os.path.splitext(os.path.basename(dat_file))[0]

    logger.info(f"Converting {animal_id}")
    os.makedirs(output_folder, exist_ok=True)

    # power spectra not needed for conversion so discarding with _
    sleep_scores, _, EEG_var, EMG_var, Temperature = readbinary_dat(dat_file)
    raw_EEG = readbinary_EEG(eeg_file)

    # Same start time Marco used in his notebook
    start_timestamp = int(datetime(2023, 1, 1, 7, 0, 0).timestamp())
    duration = 96 * 3600  # seconds

    created_files = {}

    # EEG
    eeg_path = os.path.join(output_folder, f"{animal_id}_EEG.bin")
    raw_EEG.tofile(eeg_path)  # already float32 from readbinary_EEG
    created_files['EEG'] = create_metafile_for_modality(
        binary_file=eeg_path, fs=200, no_channels=1, data_format='float32',
        start_timestamp_unix=start_timestamp, duration=duration,
        modality_type='voltage', unit='V', scale_factor=1.0,
        channel_labels=['EEG'], transmitter_id=animal_id
    )

    # EMG variance
    emg_path = os.path.join(output_folder, f"{animal_id}_EMG.bin")
    EMG_var.astype(np.float32).tofile(emg_path)
    created_files['EMG'] = create_metafile_for_modality(
        binary_file=emg_path, fs=0.25, no_channels=1, data_format='float32',
        start_timestamp_unix=start_timestamp, duration=duration,
        modality_type='variance', unit='variance', scale_factor=1.0,
        channel_labels=['EMG_variance'], transmitter_id=animal_id
    )

    # temperature
    if len(Temperature) > 0:
        temp_path = os.path.join(output_folder, f"{animal_id}_Temperature.bin")
        Temperature.astype(np.float32).tofile(temp_path)
        created_files['Temperature'] = create_metafile_for_modality(
            binary_file=temp_path, fs=0.25, no_channels=1, data_format='float32',
            start_timestamp_unix=start_timestamp, duration=duration,
            modality_type='temperature', unit='°C', scale_factor=1.0,
            channel_labels=['Temperature'], transmitter_id=animal_id
        )
    else:
        logger.info(f"  {animal_id}: temperature not recorded, skipping")

    # Sleep states
    sleep_path = os.path.join(output_folder, f"{animal_id}_SleepScore.bin")

    # mapping state characters to numbers using an ASCII lookup table.
    # score_map[ascii_code] = numeric_label, so score_map[ord('w')] = 1 etc
    score_map = np.zeros(256, dtype=np.uint8)
    score_map[ord('w')] = 1  # wake
    score_map[ord('n')] = 2  # NREM
    score_map[ord('r')] = 3  # REM
    # converting score string to a byte array, then use it as indices into score_map
    sleep_numeric = score_map[np.frombuffer(sleep_scores.encode(), dtype=np.uint8)]

    sleep_numeric.tofile(sleep_path)
    created_files['SleepScore'] = create_metafile_for_modality(
        binary_file=sleep_path, fs=0.25, no_channels=1, data_format='uint8',
        start_timestamp_unix=start_timestamp, duration=duration,
        modality_type='categorical', unit='categorical', scale_factor=1.0,
        channel_labels=['SleepScore'], transmitter_id=animal_id,
        categories={'wake': 1, 'NREM': 2, 'REM': 3, 'unknown': 0}
    )

    #logging sleep stage distribution
    for label, code in [('wake', 1), ('NREM', 2), ('REM', 3)]:
        logger.info(f"  {label}: {np.sum(sleep_numeric == code)} epochs")

    return created_files


def convert_dataset(source_folder, output_folder):
    """
    Batch convert entire FigShare dataset.
    Scans for .dat/.eeg file pairs in source_folder subfolders and converts each animal.
    """
    import glob

    # try subfolders first (e.g. data/M1EXP1/M1EXP1.dat)
    dat_files = glob.glob(os.path.join(source_folder, '*', '*.dat'))
    if not dat_files:
        dat_files = glob.glob(os.path.join(source_folder, '*.dat'))

    logger.info(f"found {len(dat_files)} animals to convert")

    converted = {}
    for dat_file in dat_files:
        eeg_file = dat_file.replace('.dat', '.eeg')
        if not os.path.exists(eeg_file):
            logger.warning(f"skipping {dat_file}: no matching .eeg file")
            continue

        animal_id = os.path.splitext(os.path.basename(dat_file))[0]
        try:
            converted[animal_id] = convert_animal_to_multimodal(
                dat_file, eeg_file, os.path.join(output_folder, animal_id), animal_id
            )
        except Exception as e:
            logger.error(f"Error converting {animal_id}: {e}")

    logger.info(f"Done: {len(converted)}/{len(dat_files)} animals converted")
    return converted


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if len(sys.argv) >= 3:
        convert_animal_to_multimodal(
            dat_file=sys.argv[1],
            eeg_file=sys.argv[2],
            output_folder=sys.argv[3] if len(sys.argv) > 3 else './converted'
        )
    else:
        print("Usage: python convert_figshare_sleep_data.py <dat_file> <eeg_file> [output_folder]")
        print("example: python convert_figshare_sleep_data.py M1EXP1.dat M1EXP1.eeg ./converted/M1EXP1")
