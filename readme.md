# Quantum Earth Mover's Distance

Code for paper "Quantum Earth Mover's Distance: A New Approach to Learning Quantum Data".

These instructions will allow you to recreate results for our paper and experiment with the quantum Wasserstein Generative Adversarial Network (qWGAN). Familiarity with Python is needed. Furthermore, access to a GPU will significantly speed up simulations.


## Layout 

There are three different folders containing code for different purposes:
- `qwgan/`: this folder contains the code used to construct a qWGAN and run various experiments using a qWGAN. All simulations performed in our paper using a qWGAN are included in this folder. Visit this folder to construct an instance of a qWGAN.
- `plotting/`: this folder contains all files used to construct figures for our paper. Requisite files in the other folders `qwgan/` and `toy_model/` must be run first before running files in this folder.
- `toy_model/`: this folder contains all code used to analyze the toy model given in section two of our paper.
 
Instructions are given in each of the subfolders to run the files contained within the folders.

## Authors

* [Bobak Kiani](https://github.com/bkiani) (MIT) 
* Giacomo De Palma (Scuola Normale Superiore)
* Milad Marvian (University of New Mexico)
* Zi-Wen Liu (Perimeter Institute)
* Seth Lloyd (MIT)
