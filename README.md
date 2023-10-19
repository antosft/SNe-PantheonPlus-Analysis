# SNe-PantheonPlus-Analysis

Our statistical analysis is based on "Apparent cosmic acceleration from type Ia supernovae" by Dam, Heinesen, Wiltshire (2017) (https://arxiv.org/pdf/1706.07236.pdf, arXiv:1706.07236) and conducted with the supernova data from the Pantheon+ catalogue. For more information on this, see Scolnic et al. (2022) (https://iopscience.iop.org/article/10.3847/1538-4357/ac8b7a, arXiv:2112.03863), Brout et al. (https://iopscience.iop.org/article/10.3847/1538-4357/ac8e04, arXiv:2202.04077) and the data release on GitHub https://github.com/PantheonPlusSH0ES/DataRelease (_not_ including the `.FITRES` calibration files our work is based on). For comparison, we also consider the JLA supernova catalogue (Betoule et al. 2014, https://www.aanda.org/articles/aa/full_html/2014/08/aa23413-14/aa23413-14.html, arXiv:1401.4064) which was used by Dam et al. (2017). The JLA dataset and covariance matrices can be downloaded from http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/568/A22 and http://supernovae.in2p3.fr/sdss_snls_jla/covmat_v6.tgz.  

The data sets and covariances used for the analysis are build from `BuildPP.py` and `BuildJLA.py`. The data sets consider redshifts in CMB frame calculated from the published heliocentric redshifts. The construction of the covariance matrix for the Pantheon+ data is described in `how_to_covariance.py` (https://github.com/antosft/SNe-PantheonPlus-Analysis/blob/main/how_to_covariance.ipynb).

The code for the Bayesian analysis (see below) requires the Multinest module (Feroz et al. 2008, https://academic.oup.com/mnras/article/398/4/1601/981502, arXiv:0809.3437) and the Python interface PyMultinest to be installed (Buchner et al. 2014, https://www.aanda.org/articles/aa/abs/2014/04/aa22971-13/aa22971-13.html, arXiv:1402.0004) as described by https://johannesbuchner.github.io/PyMultiNest/install.html#installing-the-python-module. The PyMultinest package can only be installed on linux systems, so for running the bayesian nested code, ensure you are running linux and Python 2.7.

## Build files and run the statistics code

Pipeline: `BuildPP.py` / `BuildJLA.py` > `distmod.py` > `freq_loop.py` / `bayesian_pipe.py` (calling `spline_pipe.py`) > use `parameter_freq.py` / `parameter_MLE.py` to extract the results from the output files

### `BuildPP.py`

Run as 
```
python BuildPP.py
```
Reads all `.FITRES` files from the `Pantheon/calibration_files` folder (too large to upload) as well as `fitopts_summary.csv` and `muopts_summary.csv` to get the scales for the weighting of the `FITRES` files. The covariance matrix is calculated as specified in `how_to_covariance.ipynb`. A priori, the files for P+1690 (called `PP_1690`) and P+580 (called `PP_1690jla`) are saved as outputs. By adapting `Nseeds` and `m` (number $N$ of unique supernovae included in each subsample), multiple files with the `_input.txt` and `_COVd.txt` data can be generated, including weighted ones according to the `chooseoptions` variable. If `reducelowz` is set `True`, the `joinedsample_CID+IDSURVEY.csv` file is read to adapt the fraction of low-redshift supernovae according to the JLA sample. The output is saved to `Pantheon/Build/PP_NAME_input.txt` and `Pantheon/Build/PP_NAME_COVd.txt` (`NAME` stands for the versionname variable specified in the script, e.g. `1690` for the full P+1690 or `1690random1000_0` for the first random subsample with `m = 1000`). For the random subsamples with weighted distributions, a folder `Pantheon/Build/highlow` has to be set up beforehand.

### `BuildJLA.py` and the `JLA_data` folder

Run as 
```
python BuildJLA.py
```
Reads the previously downloaded JLA data from the `JLA_data` folder. Random subsamples are constructted by the `random_JLA(...)` function at the end. The output, including the files for the common subsample JLA580, is saved to `Pantheon/Build/PP_JLA_Common/NAME_input.txt` and `Pantheon/Build/PP_JLA_Common/NAME_COVd.txt`.

### Frequentist: `freq_loop.py` and `freq_loop_marg.py`

Run as 
```
python freq_loop.py
```
after specifying `Nseeds`, `versionname`, `Nsamples`, `zcuts` and `constructdistmod` in the first lines. `Nseeds == 0` causes the script to run on a single file found at `Pantheon/Build/PP_NAME_input.txt` for `versionname = 'NAME'` instead of the Nseeds random subsamples (which would need `versionname = '1690random'` or similar). Call 
```
runfreq('path/to/input/PP_NAME_', 'path/to/output/PP_NAME_', [zcuts])
```
if working from a different script.  
An adapted version of this code that includes the marginalising procedure is available as `freq_loop_marg.py`.

### Bayesian: `bayesian_pipe.py` and `spline_pipe.py`

Run as 
```
bayesian_pipe.py modelidx zcut 0 1 2 1000 tolerance 'NAME' 'FOLDER'
```
We chose `tolerance = 1e-3`, `'FOLDER' = 'NAME/MODEL'` (for appropriate calling of files by `parameter_MLE.py`), `modelidx = 1` / `MODEL = Timescape` (timescape) or `modelidx = 2` / `MODEL = LCDM` (LCDM), `NAME` as specified in `BuildPP.py` and varying redshift cut `zcut`. The results and calculations from the MultiNest are saved with the prefix `outputpipe/FOLDER/Pantheon_modelidx_zcut_0_1_2_1000_tolerance`. This script needs the PyMultinest package and the `spline_pipe.py` file to be run successfully.

### `distmod.py`

Run as 
```
python distmod.py 'NAME'
``` 
to calculate the splined interpolation tables of the distance moduli from the `PP_NAME_input.txt` file. Alternatively, import this file and run 
```
rundistmod('path/to/PP_NAME_input.txt')
``` 
from any other script. The results are stored in `Pantheon/Build/PP_NAME_tabledL_lcdm.npy` for the standard model and `Pantheon/Build/PP_NAME_tabledL_ts.npy` for timescape.

## Read output files for plotting

### `parameter_freq.py`

Import the functions `Timescape`, `LCDM` and `Milne` to load the output from `freq_loop.py`, e.g. via  
```
import numpy as np
import pandas as pd
from parameter_freq import Timescape, LCDM, Milne
functions = {'TS': Timescape, 'LCDM': LCDM, 'Milne': Milne}
data = {}
for model in functions.keys():
    fctn = functions[model]
    data[model] = pd.DataFrame(fctn(np.loadtxt('path/to/PP_NAME_' + model + '.txt')), index=['omega', 'alpha', 'beta', 'x1', 'c', 'M'], columns=np.linspace(0.0, 0.1, 41)).T
```  
for appropriate `NAME`.

### `parameter_MLE.py`

Import the function `Parameter_Strip as ParS` to load the output from `bayesian_pipe.py` that was saved to `outputpipe/NAME/MODEL/Pantheon_modelidx_zcut_0_1_2_1000_tolerance_stats.dat` for varying `zcut` (the list of `zcuts` given by `allzcuts` in the following example) and the other parameters as specified when calling `bayesian_pipe.py`, e.g. via  
```
import numpy as np
import pandas as pd
from parameter_MLE import Parameter_Strip as ParS
allzcuts = np.linspace(0,0.1,21)
modelidcs = {'TS': 1, 'LCDM': 2}
data = {}
for model in modelidcs.keys():
    getresults = list(ParS('Pantheon_', modelidcs[model], tolerance, 'pipe/NAME', '', 13))
    omega_uncert = np.array([[x for x in lc.split(' ') if x][2] for lc in np.array(getresults[-1], dtype=str)], dtype=float)
    getresults = [np.array(r, dtype=float) for r in getresults[:-1]] + [omega_uncert]
    data[model] = pd.DataFrame(getresults, index=['Q', 'logZ', 'imp_logZ', 'alpha', 'beta', 'c', 'x1', 'omega_uncert'], columns=allzcuts[:len(getresults[0])]).T
```  
where `NAME` and `tolerance` remain to be specified. `Q` refers to `omega` for LCDM and `f_v0` for timescape (calculate `omega = 0.5*(1-fv0)*(2+fv0)` if necessary).

## Input files (within folder `Pantheon`)

### `fitopts_summary.csv`

File with information and weighting scale for `FITRES` files with varying `FITOPTS`. Called by `BuildPP.py`.

### `muopts_summary.csv`

File with information and weighting scale for `FITRES` files with varying `MUOPTS`. Called by `BuildPP.py`.

### `joinedsample_CID+IDSURVEY.csv`

IDs (in the Pantheon+ notation) and survey IDs (see https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/README for the definition) of the supernovae in the Pantheon+/JLA common subsample. Called by `BuildPP.py`.

### `.FITRES` files within `calibration_files` folder

Calibration files from folder `calibration_files` of the Pantheon+ survey for different `FITOPT` and `MUOPT` parameters. To large for upload. Read by `BuildPP.py`.

### `Build` folder

All input (`_input.txt`) and covariance (`_COVd.txt`) files go here or in subfolders of this folder. They are build by `BuildPP.py` and read by `distmod.py`, `freq_loop.py`, `bayesian_pipe.py`, etc. The outputs of `distmod.py` and `freq_loop.py` are also saved and read from here.

## Plotting example

The code for Fig. 4 from the paper is included as `fig4_sample_comparison_plot`. After constructing the files correctly, this outputs the following plot.
![Fig 4. Sample comparison plot](https://github.com/antosft/SNe-PantheonPlus-Analysis/blob/main/fig4_sample_comparison_plot_4vars.pdf "Fig 4. Sample comparison plot")
