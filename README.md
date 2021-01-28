# DeepImageDeblurring
Deep Convolutional approaches for Gaussian and Motion Blur removal.

[Project Report](DeepImageDeblurring_Report.pdf)

Keras implementation of the models and techniques discussed in the report above. These implementations has been developed by Eleonora Mancini and Marcello Simonati (guarda per taggere github) as a project work for the Deep Learning course, part of the Master's Degree in Artificial Intelligence in University of Bologna (A.A. 2019-2020).

Our network takes blurry image crops as an input and produce the corresponding sharp estimate, as in the example:

![test image](https://i.ibb.co/bJkVvYj/testimg.png)

## How to run
**Prerequisites:**
- NVIDIA GPU + CUDA CuDNN (CPU untested)
- Tensorflow 2.10 or superior
- Numpy
- Pandas
- Matplotlib

To run the scripts just clone the this repository on your workstation and execute them. Executing a script with the -h argument will pop up the help, showing you how to use them.

You can dowload our networks pretrained on CIFAR10 and REDS by clicking on the following link:

[Models and Weights](https://drive.google.com/drive/folders/17wGr5nt6D4ylh8GpIHwrbDBR2SKWbk6t?usp=sharing)

### Data
You can run the models using the CIFAR10 dataset, which will download automatically on request, or by setting a custom folder for video data. The data folder should have the following structure:

```
data
│
└───train
│   │
│   └───train_blur
│   |   │   001
│   |   │   002
│   |   │   ...
│   └───train_sharp
│       │   001
│       │   002
│       │   ...
└───val
│   │
│   └───val_blur
│   |   │   001
│   |   │   002
│   |   │   ...
│   └───val_sharp
│       │   001
│       │   002
│       │   ...
└───test
    │
    └───test_blur
    |   │   001
    |   │   002
    |   │   ...
    └───test_sharp
        │   001
        │   002
        │   ...
 ```

 Where inside the 001, 002, ... folders you should find a sequence of images belonging to the same video, numbered accordingly in order to have subsequent frames next to each other. 

This folder structure is the same adopted in the [REDS Dataset](https://seungjunnah.github.io/Datasets/reds.html) which is the dataset used for our work.

### Model Creation

You can use the create_model.py script to create a model directory and it will serialize the network to a .json file inside it. This directory will be the working directory for all other computations.

### Model Training

You can use the train_model.py script to train the model inside the specified directory. The end weights will be saved inside the model directory along with the training history in .csv format.

### Model Testing

You can test the trained model using the test_model.py script. You can also test our pretrained model by downloading the files linked above. Testing has two modes:
- evaluate: will run an evaluation on the entire test set
- batch: will generate the provided number of batches of images predicted by the network and compared with its blurred and sharp version, in a folder inside the model directory. 
