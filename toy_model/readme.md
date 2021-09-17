# Quantum Earth Mover's Distance

Code in this subfolder can be used to recreate results for the toy model (section 2). 

## Prerequisites
To run the code, first ensure that you have installed `pytorch`, `pandas`, and `numpy`. We use automatic differentiation in the `pytorch` package to perform simulations.

### Recreating results
There are two files:
- `GHZ_experiment_wass.py`: runs the toy experiment using the Earth Mover's distance as the cost metric. Running this file will output a csv file to the `csv_files/` folder named `GHZ_wass_data.csv`. This csv file will contain all data for the simulations and later can be used to create plots.
- `GHZ_experiment_innerproduct.py`: runs the toy experiment using the inner product metric as the cost metric. Running this file will output a csv file to the `csv_files/` folder named `GHZ_fidelity_adam.csv`. This csv file will contain all data for the simulations and later can be used to create plots.

Both files above use `pytorch` to perform simulations. To specify the device (e.g. use a gpu), change the `device='cpu'` line in the code to your chosen device.  E.g. `device='cuda:0'` will run on a gpu.

Final results are saved to a csv file in the `csv_files/` directory.