# SNe-PantheonPlus-Analysis

Based on "Apparent cosmic acceleration from type Ia supernovae" by Dam, Heinesen, Wiltshire (2017) https://arxiv.org/pdf/1706.07236.pdf arxiv:1706.07236 

## Build files and run the statistics code

Pipeline: `BuildPP.py` > `distmod.py` > `freq_loop.py` / `bayes_singlecut.py` (calling `loadsplines.py`) > use `parameter_freq.py` / `parameter_MLE.py` to extract the results from the output files

### `BuildPP.py`

Run as `python BuildPP.py`. Reads all calibration files as well as `fitopts_summary.csv` and `muopts_summary.csv` to get the scales for the weighting of the `FITRES` files. The covariance matrix is calculated as specified in `how_to_covariance.ipynb`. By adapting `Nseeds` and `m` (number $N$ of unique supernovae included in each subsample), multiple files with the `_input.txt` and `_COVd.txt` data can be generated. If `reducelowz` is set `True`, the `joinedsample_CID+IDSURVEY.csv` file is read to adapt the fraction of low-redshift supernovae according to the JLA sample. The output is saved to `PP_NAME_input.txt` and `PP_NAME_COVd.txt` (`NAME` stands for the versionname variable specified in the script, e.g. `1690` for the full P+1690 or `1690random1000_0` for the first random subsample with `m = 1000`).

### Frequentist: `freq_loop.py`

Run as `python freq_loop.py` after specifying `Nseeds`, `versionname`, `Nsamples`, `zcuts` and `constructdistmod` in the first lines. `Nseeds == 0` causes the script to run on a single file instead of the Nseeds random subsamples. Call `runfreq('path/to/input/PP_NAME_', 'path/to/output/PP_NAME_', [zcuts])` if working from a different script.

### Bayesian: `bayes_singlecut.py`

Run as `bayes_singlecut.py modelidx z_cut 0 1 2 nlive tolerance 'NAME'` or call `runbayes(0, modelidx, z_cut, 0, 1, 2, nlive, tolerance, 'NAME')`. We chose `nlive = 1000` and `tolerance = 1e-5`, `modelidx = 1` (timescape) or `modelidx = 2` (LCDM), `NAME` as specified in `BuildPP.py` and varying `z_cut`.

### `loadsplines.py`

Called by `bayes_singlecut.py` for loading the splined distance moduli from the `distmod.py` outputs.

### `distmod.py`

Run as `python distmodPP.py '1690'` to calculate the splined interpolation tables of the distance moduli from the `PP_1690_input.txt` file. Alternatively, import this file and run `rundistmod('path/to/PP_1690_input.txt')` from any other script. The results are stored in `PP_1690_tabledL_lcdm.npy` for the standard model and `PP_1690_tabledL_ts.npy` for timescape.

## Plotting

### `parameter_freq.py`

Import the functions `Timescape`, `LCDM` and `Milne` to load the output from `freq_loop.py`, e.g. via `pd.DataFrame(fctn(np.loadtxt('path/to/PP_NAME_' + model + '.txt')), index=['omega', 'a', 'b', 'x', 'c', 'M'], columns=np.linspace(0.0, 0.1, 41)).T` for `fctn` in `[Timescape, LCDM, Milne]` and `model` in `['TS', 'LCDM', 'Milne']`.

### `parameter_MLE.py`

Import the function `Parameter_Strip as ParS` to load the output from `bayes_singlecut.py` for varying `z_cut` (the list of `z_cuts` given by `allzcuts` in the following example) and the other parameters as specified when calling `bayes_singlecut.py`, e.g. via  
`getresults = list(ParS('Pantheon_', modelidx, tolerance, 'NAME', '', 13))`  
`omega_uncert = np.array([[x for x in lc.split(' ') if x][2] for lc in np.array(getresults[-1], dtype=str)], dtype=float)`  
`getresults = [np.array(r, dtype=float) for r in getresults[:-1]] + [omega_uncert]`  
`data = pd.DataFrame(getresults, index=['Q', 'logZ', 'imp_logZ', 'a', 'b', 'c', 'x', 'omega_uncert'], columns=allzcuts[:len(getresults[0])]).T`  
`Q` refers to `omega` for LCDM and `f_v0` for timescape (calculate `omega = 0.5*(1-fv0)*(2+fv0)` if necessary).

## Input files (within folder `Pantheon`)

### `fitopts_summary.csv`

File with information and weighting scale for `FITRES` files with varying `FITOPTS`. Called by `BuildPP.py`.

### `muopts_summary.csv`

File with information and weighting scale for `FITRES` files with varying `MUOPTS`. Called by `BuildPP.py`.

### `joinedsample_CID+IDSURVEY.csv`

IDs (in the Pantheon+ notation) of the supernovae in the Pantheon+/JLA common subsample. Called by `BuildPP.py`.

### `.FITRES` files

Calibration files from folder `calibration_files` of the Pantheon+ survey for different `FITOPT` and `MUOPT` parameters. To large for upload. Read by `BuildPP.py`.

## How to build a covariance matrix

See `how_to_covariance.py` https://github.com/antosft/SNe-PantheonPlus-Analysis/blob/main/how_to_covariance.ipynb
