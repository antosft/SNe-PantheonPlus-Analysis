# SNe-PantheonPlus-Analysis

Based on "Apparent cosmic acceleration from type Ia supernovae" by Dam, Heinesen, Wiltshire (2017) https://arxiv.org/pdf/1706.07236.pdf arxiv:1706.07236 

## Build `_input.txt` and `_COVd.txt` files



### BuildPP.py

Run as `python BuildPP.py`. Reads all files for all `FITOPT` but only `MUOPT000`. Reads `fitopts_summary.csv` to get the scales for the weighting of the `FITRES` files. Computes the sum of all statistical covariances of the fits in the different `FITOPT` files. Saves the output to `_posdef_COVd.txt` and `_posdef_input.txt` files. By specifying `mkposdef == True` at the beginning, SNe with non-positive definite covariance matrices are dropped in the output. 

### BuildPPsyst.py

Upgrade of `BuildPP.py`. Computes the fit covariance of the fits in `FITOPT000` and the systematic covariances of the other files with respect to `FITOPT000` (cf. Eq. (7) of Brout et al. 2022, arXiv:2202.04077). The saved covariance is the sum of all of these. For further information on the calculation see $\Sigma_{FITOPTS}$ below.

## Input files 

### fitopts_summary.csv

File with information and weighting scale for all `FITRES` files. Called by `BuildPP.py`.

### .FITRES files

Calibration files from folder `calibration_files` of the Pantheon+ survey for different `FITOPT` and `MUOPT` parameters. To large for upload. Read by `BuildPP.py`.

## How to build a covariance matrix

First: decide on how to calculate the `_input.txt`:
- mean of all `FITOPTS`
- `FITOPT000_MUOPT000.FITRES`  
  
The latter works better and makes more sense if we consider both `FITOPTS` and `MUOPTS` in the systematic covariance. Thus we choose `INPUT` = `FITOPT000_MUOPT000.FITRES`.  For duplicated SNe in these files we take the mean over all surveys.  
Secondly, we can calculate the different contributions to the covariance matrix. The full covariance `_COVd.txt` is then given by $$\Sigma = \Sigma_{fit} + \Sigma_{stat} + \Sigma_{dupl} + \Sigma_{FITOPTS} + \Sigma_{MUOPTS}$$

### fit covariance $\Sigma_{fit}$

1. from each line in `INPUT` (which corresponds to an observation of a supernova event, not necessarily unique events) build a blockdiagonal matrix from `COV_x0_x1` etc variables: $$\sigma^2_{SNe, survey} = \left(\begin{array}{ccc} \Delta x_0^2 & \sigma_{x_0, x_1} & \sigma_{x_0, c} \\ \sigma_{x_0, x_1} & \Delta x_1^2 & \sigma_{c, x_1} \\ \sigma_{x_0, c} & \sigma_{c, x_1} & \Delta c^2\end{array}\right)$$
2. transform this matrix to the $(m_B, x_1, c)$ space:
$$\mathbb{Cov}(m_B, x_1, c) = J \cdot \mathbb{Cov}(x_0, x_1, c) \cdot J^T$$  using $m_B = -2.5 \cdot \log_{10}x_0 + c$ and thus $J = \left(\begin{array}{ccc}
\frac{-2.5}{x_0\ln 10} & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1\\
\end{array}\right)$
3. combine covariance matrices of duplicated SNe (observed by $\nu$ different surveys): from $x_{SNe} = \frac{1}{\nu} \sum\limits_{surveys} x_{SNe, survey}$ in `INPUT` follows $$\sigma_{SNe}^2 = \sum\limits_{surveys} \left(\frac{\sigma_{SNe, survey}}{\nu}\right)^2 = \frac{1}{\nu^2} \sum\limits_{surveys} \sigma^2_{SNe, survey}$$
4. construct a block diagonal $3N \times 3N$ matrix from the $3 \times 3$ $\sigma_{SNe}^2$ blocks for each SNe $$\Sigma_{fit} = \left(\begin{array}{ccc}  \sigma_{SNe, 1}^2 & & 0 \\ & \ddots & \\ 0 & & \sigma_{SNe, N}^2\end{array}\right)$$

### statistical covariance $\Sigma_{stat}$

Brout et al. 2022 (https://iopscience.iop.org/article/10.3847/1538-4357/ac8e04/pdf) consider a statistical covariance $C_{stat}$ calculated by $$C_{stat} = \begin{cases} f \cdot \sigma_{meas}^2 + &\sigma_{scat}^2 + \sigma_{gray}^2 + \sigma_{lens}^2 + \sigma_z^2 + \sigma_{vpec}^2 & \text{same SNe and same survey} \\ &\sigma_{scat}^2 + \sigma_{gray}^2 + \sigma_{lens}^2 + \sigma_z^2 + \sigma_{vpec}^2 & \text{same SNe, different survey} \\ 0 & & \text{other off-diagonal entries}\end{cases}$$
(Equations (3), (4) and (5)). The different terms are given as follows:
- $f$ is a survey-dependent scaling to account for selection effects
- $\sigma_{meas}^2$ gives measurement uncertainty of SALT2 light-curve fit parameters
- $\sigma_{scat}^2$ describes modelled intrinsic brightness fluctuations
- $\sigma_{gray}^2$ is constant (added to $\sigma_{scat}^2$ to give $\sigma_{floor}^2$), determined after the BBC fitting process
- $\sigma_{lens}^2 = 0.055 z$
- $\sigma_z^2$ as given in the `.FITRES` files 
- $\sigma_{vpec}^2$ as given in the `.FITRES` files   
  
In our case, the measurement uncertainties ($\sigma_{meas}^2$) are given by $\Sigma_{fit}$. $\sigma_{vpec}^2$ is not relevant as we are not considering any peculiar velocities.
From the uncertainties accociated with lensing and redshift we get $$\Sigma_{stat} = \left(\begin{array}{ccc} \left(\sigma_{lens, 1}^2 + \sigma_{z, 1}^2\right)\mathbb{1}_{3\times 3} & & 0 \\ & \ddots & \\ 0 & & \left(\sigma_{lens, N}^2 + \sigma_{z, N}^2\right)\mathbb{1}_{3\times 3}\end{array}\right)$$ as a (block) diagonal $3N \times 3N$ matrix to combine with the other covariances.
To consider $f$ (possibly in $\Sigma_{fit}$), $\sigma_{scat}^2$ and $\sigma_{gray}^2$ we would need further information on the surveys.

### duplicated SNe $\Sigma_{dupl}$
For lines in `INPUT` refering to the same SNe observed by different surveys, the corresponding covariance matrix can be calculated by   
$\Sigma_{SNe} = \left(X - \bar{X}\right) \cdot \left(X - \bar{X}\right)^T$
with $X = \left(\begin{array}{c} \vdots\\m_{B, j}\\ x_{1, j}\\ c_j\\ \vdots\end{array}\right)$ where $j$ varies across the different surveys. This covariance matrix is of size $3\nu \times 3\nu$ with $\nu$ the number of surveys. As the mean over all surveys is considered in `INPUT`, this covariance matrix is reduced to a $3\times 3$ matrix for $m_B, x_1, c$ of the SNe by $$\sigma_{SNe}^2 = \left(\begin{array}{ccc} \sigma_{x_0, x_0} & \sigma_{x_0, x_1} & \sigma_{x_0, c} \\ \sigma_{x_0, x_1} & \sigma_{x_1, x_1} & \sigma_{c, x_1} \\ \sigma_{x_0, c} & \sigma_{c, x_1} & \sigma_{c, c}\end{array}\right) = \frac{1}{\nu^2} \sum\limits_{i, j \ surveys} \left(\begin{array}{ccc} \sigma_{x_0, x_0}^{ij} & \sigma_{x_0, x_1}^{ij} & \sigma_{x_0, c}^{ij} \\ \sigma_{x_0, x_1}^{ij} & \sigma_{x_1, x_1}^{ij} & \sigma_{c, x_1}^{ij} \\ \sigma_{x_0, c}^{ij} & \sigma_{c, x_1}^{ij} & \sigma_{c, c}^{ij}\end{array}\right)$$ similar to the combination of duplicated surveys in $\Sigma_{fit}$ (although the calculation was a bit easier before due to the block-diagonal structure of $\Sigma_{fit}$). Then, these blocks (one per SNe) are again combined to a block-diagonal $3N \times 3N$ matrix:
$$\Sigma_{dupl} = \left(\begin{array}{ccc}  \sigma_{SNe, 1}^2 & & 0 \\ & \ddots & \\ 0 & & \sigma_{SNe, N}^2\end{array}\right)$$

### systematic covariance from `FITOPTS` $\Sigma_{FITOPTS}$

Equation (7) from Brout et al. 2022 (https://iopscience.iop.org/article/10.3847/1538-4357/ac8e04/pdf) states $$C^{ij}_{syst} = \sum\limits_\psi \frac{\partial \Delta \mu^i_\psi}{\partial S_\psi} \frac{\partial \Delta \mu^j_\psi}{\partial S_\psi} \sigma_psi^2$$ which translates to  
`diff = scale * ((df1[VARNAME_MU] - df1[VARNAME_MUREF]) - \
                    (df2[VARNAME_MU] - df2[VARNAME_MUREF])).to_numpy()
    diff[~np.isfinite(diff)] = 0
    cov = diff[:, None] @ diff[None, :]`  
in the `create_covariance.py` file (function `get_cov_from_diff(...)`) in the SNANA GitHub repo to get the individual $\psi$ terms that are summed up later.  
In our case we choose `df2`=`INPUT` and vary `df1` across the other `FITOPT` files. We assume `df1[VARNAME_MUREF]`=`df2[VARNAME_MU]` which reduces `diff` to `scale * (df1[VARNAME_MU]- df2[VARNAME_MU])`. Furthermore, we want to consider $m_B, x_1, c$ instead of $\mu$ only. Thus, our calculation is as follows:
$$\Sigma_{FITOPTS} = \sum\limits_\psi \delta_\psi \cdot \delta_\psi^T$$
where $\psi$ varies across the different `FITOPT` files. $\delta_\psi$ is defined as $\delta_\psi = \left(\left(\begin{array}{c} \vdots\\m_{B, i}\\ x_{1, i}\\ c_i\\ \vdots\end{array}\right)_{\psi} - \left(\begin{array}{c} \vdots\\m_{B, i}\\ x_{1, i}\\ c_i\\ \vdots\end{array}\right)_{INPUT}\right) \cdot \sigma_\psi$ with $i$ enumerating the (unique) SNe in `INPUT`.

### systematic covariance from `MUOPTS` $\Sigma_{MUOPTS}$
The systematic covariance given by the `MUOPT` files can be calculated from the same formulae as $\Sigma_{FITOPTS}$ by simply varying $\psi$ across the `MUOPT` files instead of the `FITOPTS`.
