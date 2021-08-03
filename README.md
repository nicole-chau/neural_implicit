# Neural Implicit
a folder for HKU summer internship


## Updated on 31/07/2021
File dataset.py was added to show you how you can write a customized dataset.
File network.py was added to show how a 8-layer MLP can be made. (It has no dropout or other weight initialization/normalization)
File geometry.py was added to re-organize the codes; geometry related functions and classes are moved to this package/file
File basics.py was modified to use the newly added files.

## Init

FIle "basics.py" is a piece of code that you can play with. It is adapted from this repo: https://github.com/Oktosha/DeepSDF-explained

Note that "basics.py" is not the re-implementation of DeepSDF (https://github.com/facebookresearch/DeepSDF) but more similar to the paper (https://arxiv.org/abs/2009.09808) in the sense that the network is trained to overfit a single shape (but 2D in our case).


