# mlpynem

This package was created by Hugo Lourenço-Martins and Adrien Teurtrie. This package has two main goals:
- The simulation on Photo-Induced Near-field Electron Microscopy (PINEM)
- The training of neural networks for the analysis of PINEM data

## Structure of the package 

### Dataset

The purpose of the modules `pinem` and `datasets` is to create datasets that are usable (and re-usable) for the training of the neural networks. The outputs are stored in the `./datasets/` folder.

### Models

The module `models` contains the `CNNModel` and `DNNModel` objects to instanciate neural networks. The models should be save under `./models/`.

### Logs

The module `logger` is used to log the results of the neural networks in json files. The results are stored in `./logs/`.

## How to install

The neural networks can be trained faster using the GPU of your computer. However, the installation of GPU support depends a lot from computer to computer (and is not always possible). Thus we propose two install methods: with and without GPU support.

This installation guide assumes the use of anaconda.

### With GPU support

First install the libraries required for python to communicate with your GPU. For example, with a Nvidia GPU you will need to install cuda.

Then run the following commands:

```
conda create -n myenv python=3.9 -y
conda activate myenv
conda install tensorflow-gpu
conda install keras-gpu
```

Then move to where you downloaded the package and, in the same terminal, install the package:

```
pip install -e .
```

### Without GPU support

TODO : To be fully tested.

Run the following commands:
```
conda create -n myenv python=3.9 -y
conda activate myenv
conda install tensorflow
conda install keras
```

Then move to where you downloaded the package and, in the same terminal, install the package:

```
pip install -e .
```

## Use the package

We recommend the use of the notebooks. Run first the `datasets.ipynb` notebook and then run the `train_cnn.ipynb`. Then you can compare the performance of the different networks by looking at the logs.

## To do

### Major

- Re-purpose the package for the training of neural networks for time-resolved transmisson electron microscopy (TRTEM). Taking as input external simulations.
- Make tests !!
- Re-structure the package with preset neural network structure that work well for TRTEM analysis. 

### Minor

- Find a way to improve the parameters pipeline (between what is trained, what is in the datasets, etc...).
- Maybe move to wandb for logging instead of this clumsy custom logging.
