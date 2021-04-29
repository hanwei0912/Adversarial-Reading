# Adversarial Machine Learning Reading List
By Nicholas Carlini

## Preliminary Papers

### [Evasion Attacks against Machine Learning at Test Time](https://arxiv.org/pdf/1708.06131.pdf)

- well-motivated attack scenario: (1) deployed system; (2) test time.
- effective gradient-based approach
- proactive protection mechanisms that anticipate and prevent the adversarial impact: requires
(i) finding potential vulnerabilities of learning before they are exploited by the
adversary; (ii) investigating the impact of the corresponding attacks (i.e., evaluating classifier security); and (iii) devising appropriate countermeasures if an
attack is found to significantly degrade the classifier’s performance.

- Two approaches have previously addressed security issues in learning:
(1) min-max approach: assumes the learner and attacker’s loss functions are antagonistic, which yields relatively simple optimization problems
(2)  A more general game-theoretic approach applies for non-antagonistic losses;

- give the idea of the conceptions about adversary model like: Adversary’s goal,Adversary’s knowledge,Adversary’s capability,Attack scenarios(knowledge)

- in this method, their loss function has two items, (1) the discriminant function of a surrogate classifier; (2) kernel density estimator. The first term is to make sure the example successful attack the model, while the second term is to increase the probability of successful evasion. The attacker should favor attack points from densely populated regions of legitimate points. The extra component favors
attack points that imitate features of known legitimate samples. In doing so, it
reshapes the objective function and thereby biases the resulting gradient descent
towards regions where the negative class is concentrate

- this paper use the surogate for the discriminant function, which means not 100\% white-box. It also related to variance machine learning models, \eg SVM. It means it provides a framework. This work is not complicated but is a fundation work for the adversarial attacks. (2017, or earlier)

### [Intriguing properties of neural networks](https://arxiv.org/pdf/1312.6199.pdf)

- This work provides two counter-intuitive properties: (1) there is no distinction between individual high level units and
random linear combinations of high level units, according to various methods of
unit analysis. It suggests that it is the space, rather than the individual units, that
contains the semantic information in the high layers of neural networks. (2) deep neural networks learn input-output mappings that are
fairly discontinuous to a significant extent. We can cause the network to misclassify an image by applying a certain hardly perceptible perturbation, which is found
by maximizing the network’s prediction error. In addition, the specific nature of
these perturbations is not a random artifact of learning: the same perturbation can
cause a different network, that was trained on a different subset of the dataset, to
misclassify the same input.

- In this paper, the observation of adversarial perturbation is a proof to the discontinuous of input-output mappings. And in this paper, it shows the transferability of the adversarial examples already.

- In section 4.3, the measurement of upper Lipschitz constant has some relationship to my distortion measurements according the layers.

- We emphasize that we compute upper bounds: large bounds do not automatically
translate into existence of adversarial examples; however, small bounds guarantee that no such examples can appear. This suggests a simple regularization of the parameters, consisting in penalizing
each upper Lipschitz bound, which might help improve the generalisation error of the networks.

### [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf)

- Explaining: Early
attempts at explaining this phenomenon focused on nonlinearity and overfitting.
We argue instead that the primary cause of neural networks’ vulnerability to adversarial perturbation is their linear nature. This explanation is supported by new
quantitative results while giving the first explanation of the most intriguing fact
about them: their generalization across architectures and training sets. Moreover,
this view yields a simple and fast method of generating adversarial examples.

- linear explaination; FGSM; adversarial training; weight decay; transferability

- The stability of the underlying classification weights in turn results in
the stability of adversarial examples.

## Attacks

### [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/pdf/1511.07528.pdf)

- formalize the space of adversaries against DNNs, introduce a novel class of algortithms to craft adversarial sampels.
- alter a fractions of input features leading to reduced perturbation of the source inputs. (Enables adversaries to apply heuristic searches to find perturbations)
- Our understanding of how changes made to inputs affect
a DNN’s output stems from the evaluation of the forward
derivative: a matrix we introduce and define as the Jacobian
of the function learned by the DNN. The forward derivative is
used to construct adversarial saliency maps indicating input
features to include in perturbation δX in order to produce
adversarial samples inducing a certain behavior from the DNN
- Contributions: formalize the space of adversaries with respct to adversarial goal and capabilities; new class to generate adversarial examples, (forward derivatieves, adversarial saliency maps); define and measure sample distortion and source-to-target hardeness.
- Takeaways messages: (1) small input variation can leat to extreme variations of the output of the
neural network, (2) not all regions from the input domain are
conducive to find adversarial samples, and (3) the forward
derivative reduces the adversarial-sample search space.

- fully-connected networks; JSMA; success rate per source-target class pair; hardness measure H(s,t,distoriton);adversarial distance (related to non-zeros elements in adversarial saliency map)


- ideas & comments: the main difference between "forward derivative" and "gradient descent" is "forward derivative" make full use of the probability vector, while "gradeitn descent" only uses the loss function value or the probability value of the target class. Since it mentions we could use heuristic methods, it is possible to use evolutionary algorithm here also. but what is the motivation to use it to generate adversarail perturbation? one-pixel attack give one motivation, maybe we could explore the other graphic meanings....


### [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/pdf/1511.04599.pdf)

- deepFool; large-scale dataset, efficiently compute perturbations, simple yet accurate method.
- the idea is estimate the boundary with hyperplane, and then estimate the distance of the original images to each hyperplane, then choose the nearest one and then move towards this direction.
- they also study the fine-tuning using adversarial examples in MNIST and Cifar-10 on deepFool adversarial examples and FGS< adversarial examples.

### [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/pdf/1608.04644.pdf)

- CW attack

### [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://arxiv.org/abs/1802.00420)

- 'obfuscated gradients' -> 'iterative optimization-based attacks'

### [Adversarial Risk and the Dangers of Evaluating Against Weak Attacks](https://arxiv.org/pdf/1802.05666.pdf)

- abstract: We
motivate adversarial risk as an objective for
achieving models robust to worst-case inputs. We
then frame commonly used attacks and evaluation
metrics as defining a tractable surrogate objective
to the true adversarial risk. This suggests that
models may optimize this surrogate rather than
the true adversarial risk. We formalize this notion
as obscurity to an adversary, and develop tools
and heuristics for identifying obscured models
and designing transparent models. We demonstrate that this is a significant problem in practice
by repurposing gradient-free optimization techniques into adversarial attacks, which we use to
decrease the accuracy of several recently proposed
defenses to near zero.

- contributions: (1) experimentally validate the ‘security by obscurity’ nature of recently
proposed defense methods. Specifically, we show that by
using a more powerful attack method (one better able to
maximize the true adversarial risk), we can dramatically
reduce the performance of these defenses. (2) Formulation of adversarial attacks and defenses as optimizing surrogates of the true adversarial risk
(3) Developing the notion of obscurity to an adversary as
a tool for reasoning about when models optimize the
surrogate without optimizing the true adversarial risk
(4) Use of gradient-free optimization techniques to measure when models are obscured to transfer-based and
gradient-based attacks



## Transferability

### Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples

### Delving into Transferable Adversarial Examples and Black-box Attacks

### Universal adversarial perturbations

## Detecting Adversarial Examples

### On Detecting Adversarial Perturbations

### Detecting Adversarial Samples from Artifacts

### Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods

## Restricted Threat Model Attacks

### ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models

### Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models

### Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors

## Physical-World Attacks

### Adversarial examples in the physical world

### Synthesizing Robust Adversarial Examples

### Robust Physical-World Attacks on Deep Learning Models

## Verification

### Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks

### On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models

## Defenses

### Towards Deep Learning Models Resistant to Adversarial Attacks

### Certified Robustness to Adversarial Examples with Differential Privacy

### Towards the first adversarially robust neural network model on MNIST

### On Evaluating Adversarial Robustness

## Other Domains

### Adversarial Attacks on Neural Network Policies
### Audio Adversarial Examples: Targeted Attacks on Speech-to-Text
### Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples
### Adversarial examples for generative models
