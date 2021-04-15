### [Adversarial Examples Are a Natural Consequence of Test Error in Noise](https://arxiv.org/pdf/1901.10513.pdf)

#### Summary of this paper

- Estabilishing close connections between the adversarial robustness and corruption robustness research programs, they suggest that improving adversarial robustness should go hand in hand with improving performance in the presence of more general and realistice image corrutions.
- The nearby errors we can find show up at the same distance scales we would expect from a linear model with the same corruption robustness
- concentration of measure shows that a non-zero error rate in Gaussian noise logically implies the existence of small adversarial perturbations of noisy images.
- Finally, training procedures defigned to improve adversarial robustness also improve many types of corruption robustness, and training on Gaussian noise moderately improves adversarial robustness. 

### [Improving the Robustness of Deep Neural Networks via Stability Training](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45227.pdf)

- observation: small perturbations in the visual input can siginicant distort the featrue embeddings and output of a neural netowrk;
- stability traning method: triplet ranking loss; data augment by jpeg compression, thumbnail resizing, rabdin cropping.

### [Analyzing the Noise Robustness of Deep Neural Networks](https://arxiv.org/pdf/2001.09395.pdf)

#### visual analysis
1. Network-centric methods:
Network-centric methods help explore the entire network structure of a DNN, illustrating the
roles of neurons/neuron connections/layers in the training/test
process. In the pioneering work, Tzeng et al. [31] employed a DAG
visualization to illustrate the neurons and their connections. This
method can illustrate the structure of a small neural network but
suffers from severe visual clutter when visualizing state-of-the-art
DNNs. To solve this problem, Liu et al. [30] developed a scalable
visual analysis tool, CNNVis, based on clustering techniques. It
helps explore the roles of neurons in a deep CNN and diagnose
failed training processes. Wongsuphasawat et al. [32] developed a
tool with a scalable graph visualization to present the dataflow of
a DNN. To produce a legible graph visualization, they applied a
set of graph transformations that converts the low-level graph of
dataflow to the high-level structure of a DNN.
2. instance-centric methods:
These attempts aim at analyzing the learning behavior of a DNN
revealed by the instances. A widely-used method is feeding a set of
instances into a DNN and visualizing the corresponding log data,
such as the activation or the final predictions.
Rauber et al. [33] designed a compact visualization to reveal how the internal activation of training examples
evolves during a training process. They used t-SNE [38] to project
the high-dimensional activation maps of training examples in each
snapshot onto a 2D plane. The projected points are connected
by 2D trails to provide an overview of the activation during the
whole training process. The method successfully demonstrated how
different classes of instances are gradually distinguished by the
target DNN. In addition to internal activation, the final predictions
of instances can also help experts analyze the instance relationships.
For example, the tool Blocks [20] utilizes a confusion matrix to
visualize the final predictions of a large number of instances. To
reduce the visual clutter caused by a large number of instances and
classes, researchers enhanced the confusion matrix using techniques
such as non-linear color mapping and halo-based visual boosting.
3. Hybrid method:
the hybrid methods also feed the target instances
into the network and extract log data such as activation maps. The
extracted log data is often visualized in the context of the network
structure, which provides visual hints to select and explore the
data of interest, e.g., the activation in a specific layer. Visualizing
the log data in the context of network structure also helps experts
explore the data flow from the network input to the output [39].

- datapath: 
R1 - Extracting the datapaths for adversarial and normal
examples
R2 - Comparing the datapaths of adversarial and normal
examples.
R3 - Exploring datapaths at different levels
R4 - Examining how neurons contribute to each other in a
datapath



### [Robustness of classifiers: from adversarial to random noise](https://arxiv.org/pdf/1608.08967.pdf)

#### Summary of this paper

- In this paper, we propose to study a semi-random noise regime
that generalizes both the random and worst-case noise regimes. We propose
the first quantitative analysis of the robustness of nonlinear classifiers in this
general noise regime. We establish precise theoretical bounds on the robustness of
classifiers in this general regime, which depend on the curvature of the classifier’s
decision boundary. Our bounds confirm and quantify the empirical observations that
classifiers satisfying curvature constraints are robust to random noise. Moreover,
we quantify the robustness of classifiers in terms of the subspace dimension in
the semi-random noise regime, and show that our bounds remarkably interpolate
between the worst-case and random noise regimes. 

### [Adversarial vulnerability for any classifier](https://arxiv.org/abs/1802.08686)
