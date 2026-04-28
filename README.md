![PyEcog](https://raw.githubusercontent.com/KullmannLab/pyecog2/master/pyecog2/icons/banner_small.png)
# Pyecog2
Under construction.

PyEcog2 is a python software package aimed at exploring, visualizing and analysing (video) EEG telemetry data

## Installation instructions

For alpha testing:
- clone the repository to your local machine
- create a dedicated python 3.8 environment for pyecog2 (e.g. a [conda](https://www.anaconda.com/products/individual) environment)
```shell
conda create --name pyecog2 python=3.11 
```
- activate the environment with `activate pyecog2` in Windows or `source activate pyecog2` in MacOS/Linux
- run pip install with the development option :
```shell
python -m pip install -e <repository directory>
```

- To launch PyEcog you should now, from any directory, be able to call:
```shell
python -m pyecog2
```

Hopefully in the future:
```shell
pip install pyecog2
```

---

## Dataset

This project uses the **Hubbard et al. (2020)** mouse sleep dataset, doi: [10.6084/m9.figshare.12245366](https://doi.org/10.6084/m9.figshare.12245366). It covers two cohorts: EXP1 (29 mice, EEG/EMG + sleep scores) and EXP2 (9 mice, as EXP1 plus cortical temperature).

---

## Converting the FigShare Dataset

Raw files must be converted to PyEcog's `.bin`/`.meta` format before use. Run **[`Notebooks/ConvertDataFigshare.ipynb`](Notebooks/ConvertDataFigshare.ipynb)** to convert all animals.

To convert a single animal from the command line:

```shell
python -m pyecog2.convert_figshare_sleep_data <dat_file> <eeg_file> [output_folder]
```

To convert a full folder in python:

```python
from pyecog2.convert_figshare_sleep_data import convert_dataset
convert_dataset(source_folder='data/', output_folder='converted/')
```

---

## Running PyEcog2

Launch the GUI:

```shell
python -m pyecog2
```

Use **File → Load Directory** and select the converted output folder. PyEcog2 will load all modalities as separate properties within the File list to select from.

---

## Data Modalities

After loading a converted animal, you can view the following modalities:

- **EEG / EMG** — raw EEG signal at 200 Hz with EMG standard deviation on a second channel
- **Sleep Score** — Wake / NREM / REM labels at one per 4-second epoch
- **Temperature** *(EXP2 only)* — cortical temperature in °C at one per 4-second epoch
---

## Classification Notebooks

All classification work is on the `automated_classification` branch under `Notebooks/`.

| Notebook | Purpose |
|----------|---------|
| [`EDA.ipynb`](EDA.ipynb) | Exploratory Data Analysis for the Dataset |
| [`SleepClassifier.ipynb`](SleepClassifier.ipynb) | Random Forest and XGBoost on EXP1 |
| [`SVMClassifier.ipynb`](SVMClassifier.ipynb) | Linear SVM |
| [`HMMPostProcessing.ipynb`](HMMPostProcessing.ipynb) | HMM post-processing applied to RF and XGBoost probability outputs |
| [`EXP2_Temperature.ipynb`](EXP2_Temperature.ipynb) | RF on EXP2 mice with and without temperature as a feature |

---

## Unit Tests

```shell
python -m pytest UnitTestFiles/
```

Tests cover `encode_sleep_states()` in `convert_figshare_sleep_data.py` and `get_modality_info()` / `upsample_data()` in `modality_utils.py`.
