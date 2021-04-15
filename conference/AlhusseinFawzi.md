## [Alhussein Fawzi](http://www.alhusseinfawzi.info/)

## Thesis

### [Robust image classification: analysis and applications](https://infoscience.epfl.ch/record/223521/files/EPFL_TH7258.pdf)

## Review

### [A Geometric Perspective on the Robustness of Deep Networks](https://infoscience.epfl.ch/record/229872/files/spm_preprint.pdf?version=2)

- It is a review artical. Discuss the affect of : adversarial perturbations, random noises and geometric transform. 

## Attacks

### [DeepFool: a simple and accurate method to fool deep neural networks](http://arxiv.org/abs/1511.04599)

### [Universal adversarial perturbations](http://arxiv.org/abs/1610.08401)

### [Robustness of Classifiers to Universal Perturbations: A Geometric Perspective](https://openreview.net/references/pdf?id=BJX7aTpvM)

## Defenses

### [Adaptive data augmentation for image classification](https://infoscience.epfl.ch/record/218496/files/ICIP_CAMERAREADY_2715.pdf)

#### Summary of the work

- Automatic and adaptive algorithm for choosing the transformations of the samples used in data sugmentation: (1)Seek a small transformation (2) yield maximal classification loss on the transformed sample

Trust-region optimization strategy

Comments: The idea of data sugmentation is like adversarial training, they try to find the worst case, and when such kind of worst-case is found, they use it to update model parameter with a certain prosibility. But in this paper, they tried to propose more foundamential framework, they claim the derivations and algorithms developed in this paper are not specific to deep networks. And adversarial training is aimed to deep networks.

### [Empirical study of the topology and geometry of deep networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Fawzi_Empirical_Study_of_CVPR_2018_paper.pdf)

### [Robustness via curvature regularization, and vice versa](https://arxiv.org/abs/1811.09716)

#### Summary of the work

-  In this paper, we investigate the effect of adversarial training on the geometry of the classification landscape and decision boundaries. We show in particular that adversarial training leads to a significant decrease in the curvature of the loss surface with respect to inputs, leading to a drastically more "linear" behaviour of the network. Using a locally quadratic approximation, we provide theoretical evidence on the existence of a strong relation between large robustness and small curvature. To further show the importance of reduced curvature for improving the robustness, we propose a new regularizer that directly minimizes curvature of the loss surface, and leads to adversarial robustness that is on par with adversarial training. Besides being a more efficient and principled alternative to adversarial training, the proposed regularizer confirms our claims on the importance of exhibiting quasi-linear behavior in the vicinity of data points in order to achieve robustness.

### [Adversarial Robustness through Local Linearization](https://arxiv.org/pdf/1907.02610)

- In this work, we
introduce a novel regularizer that encourages the loss to behave linearly in the vicinity of the training data, thereby penalizing gradient obfuscation while encouraging
robustness

## Robustness Analysis - Noise related

### [Fundamental limits on adversarial robustness](http://www.alhusseinfawzi.info/papers/workshop_dl.pdf)

#### Summary of the work

- Distinguishable measure between the classes: Here they proposed the sphere and the boundary is a line accross the sphere. They measure the distance of noise as the radius of the sphere, while the distance of adversairal is to the boundary which is smaller than the radius.

- They anlaysis two cases: Linear classifiers and Quadratic classifiers. And they gave the upper bound to the adversarial perturbation and random uniform noise.

- Conclusion: (1) Our
result implies that in tasks involving small distinguishability, no classifier in the considered set
will be robust to adversarial perturbations, even if
a good accuracy is achieved. (2) we show
the existence of a clear distinction between the
robustness of a classifier to random noise and its
robustness to adversarial perturbations. Specifically, in high dimensions, the former is shown
to be much larger than the latter for linear classifiers

#### What I can use

- the quantities of interest: risk, robustness to adversarial perturbation (they define it as expectation of the minimim distortion), robustness to random uniform noise. The risk is related to the ground truth, accuracy, the rest is only about distortion.

### [Analysis of classifiers' robustness to adversarial perturbations](http://arxiv.org/abs/1502.02590)

#### Summary of the work

- an extersion of the "Fundamental limits on adversarial robustness", they add more analysis and experiments in different cases.

### [Robustness of classifiers: from adversarial to random noise](https://arxiv.org/abs/1608.08967)

#### Summary of the work

- Study Semi-random noise regime that generalizeds both the random and worst-case noise regimes. Quantitiative analysis of the robustness of nonlinear classifier in this general noise regime. 

### [Robustness of classifiers to uniform \ell_p and Gaussian noise](https://arxiv.org/abs/1802.07971)

- Add the noise drawn uniformly from the \ell_p ball and Gaussian noise

### [Adversarial vulnerability for any classifier](http://www.alhusseinfawzi.info/papers/fawzi18_adversarial.pdf)

#### Summary of the work

- Assumption: the data is generated with a smooth generative model. (Now the case is make the "noise" photorealism)


## Robustness Analysis - Geometric related

### [Manitest: Are classifiers really invariant?](http://infoscience.epfl.ch/record/210209/files/bmvc_paper.pdf)

#### Summary of the work

- Rigorous and systematic approach for quantifying the invariance to geometric transformation of any classifier

They represent this problem by defining the transformation as a Lie group consisting of geomatric transformations.

- Conclusion: (1)we
were able to convert the problem of assessing the classifier’s invariance to that of computing
geodesic distances. (2) we quantified the increasing invariance of CNNs with
depth, and highlighted the importance of data augmentation for learning invariance from
data.

### [Robustness of Classifiers to Universal Perturbations: A Geometric Perspective](https://openreview.net/references/pdf?id=BJX7aTpvM)

#### Summary of the work

- Analysis the geometric view of the classifier boundary 
 In this paper, we provide a quantitative
analysis of the robustness of classifiers to universal perturbations, and draw a formal
link between the robustness to universal perturbations, and the geometry of the
decision boundary. Specifically, we establish theoretical bounds on the robustness
of classifiers under two decision boundary models (flat and curved models). We
show in particular that the robustness of deep networks to universal perturbations
is driven by a key property of their curvature: there exist shared directions along
which the decision boundary of deep networks is systematically positively curved.
Under such conditions, we prove the existence of small universal perturbations.
Our analysis further provides a novel geometric method for computing universal
perturbations, in addition to explaining their properties.

### [Empirical study of the topology and geometry of deep networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Fawzi_Empirical_Study_of_CVPR_2018_paper.pdf)

- find a patch between two data poisnt.

We specifically study the topology of classification
regions created by deep networks, as well as their associated
decision boundary. Through a systematic empirical study, we
show that state-of-the-art deep nets learn connected classification regions, and that the decision boundary in the vicinity
of datapoints is flat along most directions. We further draw
an essential connection between two seemingly unrelated
properties of deep networks: their sensitivity to additive perturbations of the inputs, and the curvature of their decision
boundary. The directions where the decision boundary is
curved in fact characterize the directions to which the classifier is the most vulnerable. We finally leverage a fundamental
asymmetry in the curvature of the decision boundary of deep
nets, and propose a method to discriminate between original images, and images perturbed with small adversarial
examples. We show the effectiveness of this purely geometric
approach for detecting small adversarial perturbations in
images, and for recovering the labels of perturbed images

### [Are labels required for improving adversarial robustness?](https://arxiv.org/pdf/1905.13725)

Recent work has uncovered the interesting (and somewhat surprising) finding that
training models to be invariant to adversarial perturbations requires substantially
larger datasets than those required for standard classification. This result is a key
hurdle in the deployment of robust machine learning models in many real world
applications where labeled data is expensive. Our main insight is that unlabeled
data can be a competitive alternative to labeled data for training adversarially robust
models. Theoretically, we show that in a simple statistical setting, the sample
complexity for learning an adversarially robust model from unlabeled data matches
the fully supervised case up to constant factors. On standard datasets like CIFAR10, a simple Unsupervised Adversarial Training (UAT) approach using unlabeled
data improves robust accuracy by 21.7% over using 4K supervised examples alone,
and captures over 95% of the improvement from the same number of labeled
examples. Finally, we report an improvement of 4% over the previous state-of-theart on CIFAR-10 against the strongest known attack by using additional unlabeled
data from the uncurated 80 Million Tiny Images dataset. This demonstrates that
our finding extends as well to the more realistic case where unlabeled data is also
uncurated, therefore opening a new avenue for improving adversarial training.
