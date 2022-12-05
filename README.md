# SNe-PantheonPlus-Analysis

Based on "Apparent cosmic acceleration from type Ia supernovae" by Dam, Heinesen, Wiltshire (2017) https://arxiv.org/pdf/1706.07236.pdf arxiv:1706.07236 

## BuildPP.py

Run as `python BuildPP.py f m` with f the FITOPT parameter and m the MUOPT parameter of the file to be read. The output file's names will be of the form `PP_full_Ff_Mm_` (with f and m as before) continued by `COVd` or `input`.  
Example: `BuildPP.py 0 0` reads `FITOPT000_MUOPT000.FITRES` and saves to `PP_full_F000_M000_COVd.txt` and `PP_full_F000_M000_input.txt`
