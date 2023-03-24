# SNe-PantheonPlus-Analysis

Our statistical analysis is based on "Apparent cosmic acceleration from type Ia supernovae" by Dam, Heinesen, Wiltshire (2017) (https://arxiv.org/pdf/1706.07236.pdf, arXiv:1706.07236) and conducted with the supernova data from the Pantheon+ catalogue. For more information on this, see Scolnic et al. (2022) (https://iopscience.iop.org/article/10.3847/1538-4357/ac8b7a, arXiv:2112.03863), Brout et al. (https://iopscience.iop.org/article/10.3847/1538-4357/ac8e04, arXiv:2202.04077) and the data release on GitHub https://github.com/PantheonPlusSH0ES/DataRelease (_not_ including the `.FITRES` calibration files our work is based on). For comparison, we also consider the JLA supernova catalogue (Betoule et al. 2014, https://www.aanda.org/articles/aa/full_html/2014/08/aa23413-14/aa23413-14.html, arXiv:1401.4064) which was used by Dam et al. (2017). The JLA dataset and covariance matrices can be downloaded from http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/568/A22 and http://supernovae.in2p3.fr/sdss_snls_jla/covmat_v6.tgz.  

The data sets and covariances used for the analysis are build from `BuildPP.py` and `BuildJLA.py`. The data sets consider redshifts in CMB frame calculated from the published heliocentric redshifts. The construction of the covariance matrix for the Pantheon+ data is described in `how_to_covariance.py` (https://github.com/antosft/SNe-PantheonPlus-Analysis/blob/main/how_to_covariance.ipynb).

The code for the Bayesian analysis (see below) requires the Multinest module (Feroz et al. 2008, https://academic.oup.com/mnras/article/398/4/1601/981502, arXiv:0809.3437) and the Python interface PyMultinest to be installed (Buchner et al. 2014, https://www.aanda.org/articles/aa/abs/2014/04/aa22971-13/aa22971-13.html, arXiv:1402.0004) as described by https://johannesbuchner.github.io/PyMultiNest/install.html#installing-the-python-module. The Pymultinest can only be installed on linux systems, so for running the bayesian nested code, ensure you are running linux and Python 2.7.

## Build files and run the statistics code

Pipeline: `BuildPP.py` > `distmod.py` > `freq_loop.py` / `bayes_singlecut.py` (calling `loadsplines.py`) > use `parameter_freq.py` / `parameter_MLE.py` to extract the results from the output files

### `BuildPP.py`

Run as `python BuildPP.py`. Reads all `.FITRES` files from the `Pantheon/calibration_files` folder (too large to upload) as well as `fitopts_summary.csv` and `muopts_summary.csv` to get the scales for the weighting of the `FITRES` files. The covariance matrix is calculated as specified in `how_to_covariance.ipynb`. By adapting `Nseeds` and `m` (number $N$ of unique supernovae included in each subsample), multiple files with the `_input.txt` and `_COVd.txt` data can be generated. If `reducelowz` is set `True`, the `joinedsample_CID+IDSURVEY.csv` file is read to adapt the fraction of low-redshift supernovae according to the JLA sample. The output is saved to `Pantheon/Build/PP_NAME_input.txt` and `Pantheon/Build/PP_NAME_COVd.txt` (`NAME` stands for the versionname variable specified in the script, e.g. `1690` for the full P+1690 or `1690random1000_0` for the first random subsample with `m = 1000`).

### Frequentist: `freq_loop.py`

Run as `python freq_loop.py` after specifying `Nseeds`, `versionname`, `Nsamples`, `zcuts` and `constructdistmod` in the first lines. `Nseeds == 0` causes the script to run on a single file found at `Pantheon/Build/PP_NAME_input.txt` for `versionname = 'NAME'` instead of the Nseeds random subsamples (which would need `versionname = '1690random'` or similar). Call `runfreq('path/to/input/PP_NAME_', 'path/to/output/PP_NAME_', [zcuts])` if working from a different script.

### Bayesian: `bayes_singlecut.py`

Run as `bayes_singlecut.py modelidx z_cut 0 1 2 nlive tolerance 'NAME'` or call `runbayes(0, modelidx, z_cut, 0, 1, 2, nlive, tolerance, 'NAME')`. We chose `nlive = 1000` and `tolerance = 1e-5`, `modelidx = 1` (timescape) or `modelidx = 2` (LCDM), `NAME` as specified in `BuildPP.py` and varying `z_cut`. This script needs the PyMultinest package to be run successfully.

### `loadsplines.py`

Called by `bayes_singlecut.py` for loading the splined distance moduli from the `distmod.py` outputs.

### `distmod.py`

Run as `python distmodPP.py 'NAME'` to calculate the splined interpolation tables of the distance moduli from the `PP_NAME_input.txt` file. Alternatively, import this file and run `rundistmod('path/to/PP_NAME_input.txt')` from any other script. The results are stored in `Pantheon/Build/PP_NAME_tabledL_lcdm.npy` for the standard model and `Pantheon/Build/PP_NAME_tabledL_ts.npy` for timescape.

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

IDs (in the Pantheon+ notation) and survey IDs (see https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/README for the definition) of the supernovae in the Pantheon+/JLA common subsample. Called by `BuildPP.py`.

### `.FITRES` files

Calibration files from folder `calibration_files` of the Pantheon+ survey for different `FITOPT` and `MUOPT` parameters. To large for upload. Read by `BuildPP.py`.
