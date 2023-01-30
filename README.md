# SNe-PantheonPlus-Analysis

Based on "Apparent cosmic acceleration from type Ia supernovae" by Dam, Heinesen, Wiltshire (2017) https://arxiv.org/pdf/1706.07236.pdf arxiv:1706.07236 

## BuildPP.py

Run as `python BuildPP.py`. Reads all files for all `FITOPT` but only `MUOPT000`. Reads `fitopts_summary.csv` to get the scales for the weighting of the `FITRES` files. Computes the sum of all statistical covariances of the fits in the different `FITOPT` files. Saves the output to `_posdef_COVd.txt` and `_posdef_input.txt` files. By specifying `mkposdef == True` at the beginning, SNe with non-positive definite covariance matrices are dropped in the output. 

## BuildPPsyst.py

Run as `python BuildPP.py`. Reads all files for all `FITOPT` but only `MUOPT000`. Reads `fitopts_summary.csv` to get the scales for the weighting of the `FITRES` files. Computes the statistical covariance of the fits in `FITOPT000` and the systematic covariances of the other files with respect to `FITOPT000` (cf. Eq. (7) of Brout et al. 2022, arXiv:2202.04077). The saved covariance is the sum of all of these. Saves the output to `_COVd.txt` and `_input.txt` files. By specifying `mkposdef == True` at the beginning, SNe with non-positive definite covariance matrices are dropped in the output. 

### fitopts_summary.csv

File with information and weighting scale for all `FITRES` files. Called by `BuildPP.py`.

### .FITRES files

Calibration files from folder `calibration_files` of the Pantheon+ survey for different `FITOPT` and `MUOPT` parameters. To large for upload. Read by `BuildPP.py`.
