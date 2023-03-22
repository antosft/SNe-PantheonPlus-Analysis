# SNe-PantheonPlus-Analysis

Based on "Apparent cosmic acceleration from type Ia supernovae" by Dam, Heinesen, Wiltshire (2017) https://arxiv.org/pdf/1706.07236.pdf arxiv:1706.07236 

## Build files and run the statistics code

### `BuildPP.py`

Run as `python BuildPP.py`. Reads all calibration files as well as `fitopts_summary.csv` and `muopts_summary.csv` to get the scales for the weighting of the `FITRES` files. The covariance matrix is calculated as specified in `how_to_covariance.ipynb`. By adapting `Nseeds` and `m` (number $N$ of unique supernovae included in each subsample), multiple files with the `_input.txt` and `_COVd.txt` data can be generated. If `reducelowz` is set `True`, the `joinedsample_CID+IDSURVEY.csv` file is read to adapt the fraction of low-redshift supernovae according to the JLA sample. The output is saved to `PP_1690_input.txt` and `PP_1690_COVd.txt` (`1690` is replaced when considering the random subsamples).

### Frequentist: `freq_loop.py`

Run as `python freq_loop.py` after specifying `Nseeds`, `versionname`, `Nsamples`, `zcuts` and `constructdistmod` in the first lines. `Nseeds == 0` causes the script to run on a single file instead of the Nseeds random subsamples. Call `runfreq(path/to/input/, path/to/output/, [zcuts])` if working from a different script.

### Bayesian: `bayes_singlecut.py`

Run as `bayes_singlecut.py model z_cut 0 1 2 nlive tolerance` or call `runbayes(0, model z_cut, 0, 1, 2, nlive, tolerance)`. We chose `model = 1` (timescape) or `model = 2` ($\Lambda$CDM), varying `z_cut`, `nlive = 1000` and `tolerance = 1e-5`.

### `loadsplines.py`

Called by `bayes_singlecut.py` for loading the splined distance moduli from the `distmod.py` outputs.

### `distmod.py`

Run as `python distmodPP.py '1690'` to calculate the splined interpolation tables of the distance moduli from the `PP_1690_input.txt` file. Alternatively, import this file and run `rundistmod('path/to/PP_1690_input.txt')` from any other script. The results are stored in `PP_1690_tabledL_lcdm.npy` for the standard model and `PP_1690_tabledL_ts.npy` for timescape.

## Plotting

## Input files 

### `fitopts_summary.csv`

File with information and weighting scale for `FITRES` files with varying `FITOPTS`. Called by `BuildPP.py`.

### `muopts_summary.csv`

File with information and weighting scale for `FITRES` files with varying `MUOPTS`. Called by `BuildPP.py`.

### `joinedsample_CID+IDSURVEY.csv`

IDs (in the Pantheon+ notation) of the supernovae in the Pantheon+/JLA common subsample. Called by `BuildPP.py`.

### .FITRES files

Calibration files from folder `calibration_files` of the Pantheon+ survey for different `FITOPT` and `MUOPT` parameters. To large for upload. Read by `BuildPP.py`.

## How to build a covariance matrix

See `how_to_covariance.py` https://github.com/antosft/SNe-PantheonPlus-Analysis/blob/main/how_to_covariance.ipynb
