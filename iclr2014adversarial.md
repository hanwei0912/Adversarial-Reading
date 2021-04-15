### ICLR 2014 :

### [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)

* First paper observed adversarial phenomena.

#### Two Conter-intuitive properties
1. No distinction between individual high level units and random linear combinations of high level units;
2. Input-output mapping are fairly discontinuous to a significant extent. Smoothness assumpution does not hold.

####  Findings might be summarized as follows
1. Certain dimensions of the each layer reflects different semantics of data. (This is a well-known fact to this date therefore I skip this to discuss more)
2. Adversarial instances are general to different models and datasets.
3. Adversarial instances are more significant to higher layers of the networks.
4. Auto-Encoders are more resilient to adversarial instances.

#### Generate the adversarial examples

__Constructive way to generate adversarial examples__

* Box-constrained optimization problem
* Target: Misclassification
* Box-constrained L-BFGS
* "Minimimum distortion" function D

__Classifier model used__

* Simple linear (softmax) classifier without hidden units
* Simple sigmoidal neural network with two hidden layers and a classifier
* A model consists of single layer sparse autoencoder with sigmoid actions and 400 nodes with a Softmax classifier