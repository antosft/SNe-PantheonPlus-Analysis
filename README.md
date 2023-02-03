# SNe-PantheonPlus-Analysis

Based on "Apparent cosmic acceleration from type Ia supernovae" by Dam, Heinesen, Wiltshire (2017) https://arxiv.org/pdf/1706.07236.pdf arxiv:1706.07236 

## Build `_input.txt` and `_COVd.txt` files



### BuildPP.py

Run as `python BuildPP.py`. Reads all files for all `FITOPT` but only `MUOPT000`. Reads `fitopts_summary.csv` to get the scales for the weighting of the `FITRES` files. Computes the sum of all statistical covariances of the fits in the different `FITOPT` files. Saves the output to `_posdef_COVd.txt` and `_posdef_input.txt` files. By specifying `mkposdef = True` at the beginning, SNe with non-positive definite covariance matrices are dropped in the output. 

### BuildPPsyst.py

Upgrade of `BuildPP.py`. $\Sigma_{fit}$ (as in `BuildPP.py`, cf. `how_to_covariance.py`) now considering `FITOPT000` only. Other changes are focused on the systematic covariances of the other `FITOPT` files with respect to `FITOPT000` (cf. Eq. (7) of Brout et al. 2022, arXiv:2202.04077). The saved covariance includes the sum of all of these. For further information on the calculation see $\Sigma_{FITOPTS}$ in `how_to_covariance.py`.

### BuildPPdupl.py

Upgrade of `BuildPPsyst.py`. Account for covariances of duplicated SNe, as explained under $\Sigma_{dupl}$ in `how_to_covariance.py`.

### BuildPPmu.py

Upgrade of `BuildPP.py`. Changes are focused on the systematic covariances of the other `MUOPT` files with respect to `MUOPT000` (cf. Eq. (7) of Brout et al. 2022, arXiv:2202.04077). The saved covariance includes the sum of all of these. For further information on the calculation see $\Sigma_{MUOPTS}$ in `how_to_covariance.py`.

### BuildPPstat.py

Upgrade of `BuildPPmu.py`. Account for statistical covariances in terms of $\sigma_z^2$ and $\sigma_{lens}^2$, as explained under $\Sigma_{stat}$ in `how_to_covariance.py`. By specifying `ewByFile = False` at the beginning, SNe the eigenvalues of the full covariance matrix are considered instead of the eigenvalues of the blocks $\Sigma_{fit}$. Keep `ewByFile = True` to ensure a semi-positive covariance matrix.

## Input files 

### fitopts_summary.csv

File with information and weighting scale for all `FITRES` files. Called by `BuildPP.py`.

### .FITRES files

Calibration files from folder `calibration_files` of the Pantheon+ survey for different `FITOPT` and `MUOPT` parameters. To large for upload. Read by `BuildPP.py`.

## How to build a covariance matrix

See `how_to_covariance.py` https://github.com/antosft/SNe-PantheonPlus-Analysis/blob/main/how_to_covariance.ipynb
