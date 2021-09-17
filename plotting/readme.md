# Quantum Earth Mover's Distance

Use the code in this directory to construct figures that we show in our paper. Requisite files in the other folders `qwgan/` and `toy_model/` must be run first before running files in this folder. 

There are three different files used to create figures for different parts of the paper:
- `plot_toy_experiment.py`: this file recreates figure 1a of our paper. See the `../toy_model/` directory to find the requisite code to run before creating these figures.
- `plot_losses.py`: this file can be used to recreate all figures showing how the loss function changes over the course of optimization. See the `../qwgan/` directory to find the requisite code to run before creating these figures.
- `plot_gradients.py`: this file can be used to plot a comparison of the gradients of the qWGAN with a more conventional quantum GAN. See the `../qwgan/` directory to find the requisite code to run before creating these figures.
 