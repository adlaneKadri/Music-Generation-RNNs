#  What is this project about?
In this project we are focusing on generating music automatically using Recurrent Neural Network(RNN).
* Let's break the project program into 3 major part:
  * Data preparation 
    - recover the notes played in each midi file
    - data processing
  * Training
  * predection 


## Requirements

In the following software and hardware list, you can run the code file in this repository.

| Software  | OS  |
| ------------------------------------ | ----------------------------------- |
| Music21, Anaconda Package Python 2.x/3.x, TensorFlow, Keras, numpy, glob | Ubuntu 16.04 or greater |

* Use `pip` to install this packages:

E.g.

```
pip install Music21
```
you’ll need to make sure you have `pip` available. You can check this by running

```
pip --version
```

if you don't have `pip`, i suggest this [link](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/) to install it

## Training

To train the network you run **RNN.py**.

E.g.

```
python RNN.py
```

