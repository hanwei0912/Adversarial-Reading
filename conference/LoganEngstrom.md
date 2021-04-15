## [Logan Engstrom](http://loganengstrom.com/)

### Analysis

#### [Adversarial Robustness as a Prior for Learned Representations](https://arxiv.org/pdf/1906.00945.pdf)

- abstract: In this work, we show that robust optimization can be re-cast as a tool for
enforcing priors on the features learned by deep neural networks. It turns out that representations learned
by robust models address the aforementioned shortcomings and make significant progress towards learning a high-level encoding of inputs. In particular, these representations are approximately invertible, while
allowing for direct visualization and manipulation of salient input features. More broadly, our results indicate adversarial robustness as a promising avenue for improving learned representations. 

- findings: (1) *Representation inversion*: robust representations are approximately invertible—that is, they provide a high-level embedding of the input such that images with similar robust representations are semantically similar, and the salient features of
an image are easily recoverable from its robust feature representation. This property also naturally
enables feature interpolation between arbitrary inputs. (2) *Simple feature visualization*: Direct maximization of the coordinates of robust representations suffices to visualize easily recognizable features of the model. This is again a significant
departure from standard models where (a) without explicit regularization at visualization time, feature visualization often produces unintelligible results; and (b) even with regularization, visualized
features in the representation layer are scarcely human-recognizeable. (3) *Feature manipulation*: Through the aforementioned direct feature visualization property, robust representations enable the addition of specific features to images through direct first-order optimization

- Thus, the existence of these image pairs (and similar phenomena observed by prior work [Jac+19]) lays bare
a misalignment between the notion of distance induced via the features learned by current deep networks,
and the notion of distance as perceived by humans.

- *adversarial robustness as a prior*:In what follows, we will explore the effect of the prior induced by adversarial robustness on models’
learned representations, and demonstrate that representations learned by adversarially robust models are
better behaved, and do in fact seem to use features that are more human-understandable.

- *Inverting robust representations*: regard the process of generate adversarial perturbation as recovering an image that maps to the desired target representation. "Interpolation between arbitrary inputs": linearly interpolate between two representations in representation space, then use the inversion procedure to get images corresponding to the interpolate representations.

- *Direct feature visualization*: find the perturbation which maximize a specific feature(component) in the representation. works in adversarial trained networks but not working for the standard networks (meaningless to humans)

#### [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175)

- *non-robust features*:  features (derived from patterns in the data distribution) that
are highly predictive, yet brittle and (thus) incomprehensible to humans. After capturing these features
within a theoretical framework, we establish their widespread existence in standard datasets. And this also suggest an explanation for adversarial transferability. 

- Adversarial vulnerability is a direct result of our models’ sensitivity to well-generalizing features in the data.

- Able to construct: (1) *A robustified version for robust classification*: We demonstrate that it is possible to
effectively remove non-robust features from a dataset. Concretely, we create a training set (semantically similar to the original) on which standard training yields good robust accuracy on the original,
unmodified test set. This finding establishes that adversarial vulnerability is not necessarily tied to the
standard training framework, but is also a property of the dataset. (2) *A non-robust version for standard classification*: We are also able to construct a
training dataset for which the inputs are nearly identical to the originals, but all appear incorrectly
labeled. In fact, the inputs in the new training set are associated to their labels only through small
adversarial perturbations (and hence utilize only non-robust features). Despite the lack of any predictive
human-visible information, training on this dataset yields good accuracy on the original, unmodified
test set. This demonstrates that adversarial perturbations can arise from flipping features in the data
that are useful for classification of correct inputs (hence not being purely aberrations)

- $\phi$-useful features;$\gamma$-robustly useful features; useful, non-robust feature.

-  Transferability can arise from non-robust features

- A theoretical framework for studying (Non)-Robust Features: The adversarial vulnerability can be explicitly expressed as a difference between the inherent data
metric and the L2 metric. Robust learning corresponds exactly to learning a combination of these two metrics. The gradients of adversarially trained models align better with the adversary’s metric.

- we cast the phenomenon of adversarial examples as a natural consequence of the presence of
highly predictive but non-robust features in standard ML datasets. We provide support for this hypothesis byexplicitly disentangling robust and non-robust features in standard datasets, as well as showing that nonrobust features alone are sufficient for good generalization. Finally, we study these phenomena in more
detail in a theoretical setting where we can rigorously study adversarial vulnerability, robust training, and
gradient alignment.
Our findings prompt us to view adversarial examples as a fundamentally human phenomenon. In particular, we should not be surprised that classifiers exploit highly predictive features that happen to be
non-robust under a human-selected notion of similarity, given such features exist in real-world datasets.
In the same manner, from the perspective of interpretability, as long as models rely on these non-robust
features, we cannot expect to have model explanations that are both human-meaningful and faithful to
the models themselves. Overall, attaining models that are robust and interpretable will require explicitly
encoding human priors into the training process.



#### [Do Adversarially Robust ImageNet Models Transfer Better?](https://arxiv.org/abs/2007.08489)

- better pre-trained models yield better transfer results, suggesting that initial accuracy is a key aspect of transfer learning performance
- we find that adversarially robust models, while less accurate, often perform better than their standard-trained counterparts when used for transfer learning. Specifically, we focus on adversarially robust ImageNet classifiers, and show that they yield improved accuracy on a standard suite of downstream classification tasks. Further analysis uncovers more differences between robust and standard models in the context of transfer learning
- fixed-feature？？？


#### [Identifying Statistical Bias in Dataset Replication](https://arxiv.org/abs/2005.09619)
- In this work, we present unintuitive yet significant ways in which standard approaches to dataset replication introduce statistical bias, skewing the resulting observations.

#### [Implementation Matters in Deep Policy Gradient Algorithms](https://arxiv.org/abs/2005.12729)
- We study the roots of algorithmic progress in deep policy gradient algorithms
through a case study on two popular algorithms: Proximal Policy Optimization
(PPO) and Trust Region Policy Optimization (TRPO). Specifically, we investigate
the consequences of “code-level optimizations:” algorithm augmentations found
only in implementations or described as auxiliary details to the core algorithm.
Seemingly of secondary importance, such optimizations turn out to have a major
impact on agent behavior. Our results show that they (a) are responsible for most
of PPO’s gain in cumulative reward over TRPO, and (b) fundamentally change
how RL methods function. These insights show the difficulty and importance of
attributing performance gains in deep reinforcement learning.



