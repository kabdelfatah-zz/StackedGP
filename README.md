## StackedGP Models

This is a python for the StackedGP modeling framework. The main applications for StackedGP framework are to integrate different datasets through model composition, enhance predictions of quantities of interest through a cascade of intermediate predictions, and to propagate uncertainties through emulated dynamical systems driven by uncertain forcing variables. 

Kareem Abdelfatah, Junshu Bao, Gabriel Terejanu (2017). Environmental Modeling Framework using Stacked Gaussian Processes. [arXiv:1612.02897v2](https://arxiv.org/abs/1612.02897v2)

## Dependencies

This software is dependent on the following packages: 
* numpy, 
* scipy, 
* sklearn.preprocessing, 
* GPy: https://github.com/SheffieldML/GPy

## Contents

This package contains the core classes for StackedGP in "stackedgp_src" directory. It also contains the real data and applications discussed in *Kareem Abdelfatah, Junshu Bao, Gabriel Terejanu (2017). Environmental Modeling Framework using Stacked Gaussian Processes. arXiv:1612.02897v2*

## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.