# Training a neural network as implicit representation of a variety of 2D shapes
University of Hong Kong - Computer Graphics Research Intern Project 2021

This is a reimplementation of features of the DeepSDF (https://github.com/facebookresearch/DeepSDF) neural network to train a neural network as implicit representation of a variety of 2D shapes with the addition of latent vectors. The code also computes an affinity matrix for the latent vectors to compare the similarity of latent vectors generated through training for shapes with similar features.

### basics.py
Defines the 2D shapes to be used for training and implements the main training process. The network is trained for 1000 epochs and the loss at each epoch is computed as follows:

![equation](https://latex.codecogs.com/png.latex?L%28f_%5Ctheta%28x%29%2C%20s%29%20%3D%20%7C%5Ctext%7Bclamp%7D%28f_%5Ctheta%28x%29%2C%20%5Cdelta%29%20-%20%5Ctext%7Bclamp%7D%28s%2C%20%5Cdelta%29%7C)

where 

![equation](https://latex.codecogs.com/png.latex?%5Ctext%7Bclamp%7D%28x%2C%20%5Cdelta%29%20%3A%3D%20min%28%5Cdelta%2C%20max%28-%5Cdelta%2C%20x%29%29%2C%20%5C%3B%5C%3B%5Cdelta%20%3D%200.1)

"basics.py" is adapted from this repo: https://github.com/Oktosha/DeepSDF-explained

### network.py
The neural network contains the following features:
- 8 fully connected layers
    - Internal layers: 512 dimensional with ReLU non-linearities
    - Dropout with probability 0.2
- Training with weight normalization
- Adam optimizer (learning rate = 1e-5)

### dataset.py
Samples surface points from various 2D shapes (circle, polygon) to create the training dataset

### geometry.py
Creates various 2D shapes (circle, polygon) and plots their SDF using a color map
