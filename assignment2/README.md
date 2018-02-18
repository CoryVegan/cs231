# There are a lot of files here...what am I looking at?
This assignment contains 3 types of files:
  1. `*.ipynb`: This is where all the assignment problems, as well as most of my solutions, live.
  2. `cs231/`: This is where all the helper python files live. Some of the notebooks require use to modify/implement functions, within this directory, as part of our solution.  
  3. `assignment2.md`: This is the original assignment handout.

# More on each of the ipynbs
The ipynbs are meant to be completed in the following order:
  1. `FullyConnectedNets.ipynb`: Implement a **Neural Network** using a **modular design**, allowing use to build networks of **arbitrary depth**. Also, implement several optimizers: **Stochastic Gradient Descent with momentum**, **RMSprop** and **Adam**. We then use these tools to build the best model we can on the **CIFAR10** dataset.
  2. `BatchNormalization.ipynb`: Implement **Batch Normalization** and explore its effects on model training and weight scale initialization.
  3. `Dropout.ipynb`: Implement **Dropout** and explore its effects on model generalization.
  4. `ConvolutionalNetworks.ipynb`: Implement a **Convolutional Neural Network** with **Spatial Batch Normalization** and **Max Pooling**. Then, train the best network we can on the **CIFAR10** dataset.
