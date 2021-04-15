## Attack

### [SmoothFool: An Efficient Framework for Computing Smooth Adversarial Perturbations](https://openaccess.thecvf.com/content_WACV_2020/papers/Dabouei_SmoothFool_An_Efficient_Framework_for_Computing_Smooth_Adversarial_Perturbations_WACV_2020_paper.pdf)

They use low-pass filter g and estimate w using the first order Taylor expansion of f at $x_p$

### [A principled approach for generating adversarial images under non-smooth dissimilarity metrics](http://proceedings.mlr.press/v108/pooladian20a/pooladian20a.pdf)

 In this work,
we propose an attack methodology not only
for cases where the perturbations are measured by lp norms, but in fact any adversarial dissimilarity metric with a closed proximal
form. This includes, but is not limited to,
l1, l2, and l∞ perturbations; the l0 counting
“norm” (i.e. true sparseness); and the total
variation seminorm, which is a (non-lp) convolutional dissimilarity measuring local pixel
changes. Our approach is a natural extension of a recent adversarial attack method,
and eliminates the differentiability requirement of the metric. We demonstrate our
algorithm, ProxLogBarrier, on the MNIST,
CIFAR10, and ImageNet-1k datasets. We
consider undefended and defended models,
and show that our algorithm easily transfers to various datasets. We observe that
ProxLogBarrier outperforms a host of modern adversarial attacks specialized for the l0
case. Moreover, by altering images in the total variation seminorm, we shed light on a
new class of perturbations that exploit neighboring pixel information.

### [SPATIALLY TRANSFORMED ADVERSARIAL EXAMPLES](https://arxiv.org/pdf/1801.02612.pdf)

in this paper
we will instead focus on a different type of perturbation, namely spatial transformation, as opposed to manipulating the pixel values directly as in prior works.
Perturbations generated through spatial transformation could result in large Lp
distance measures, but our extensive experiments show that such spatially transformed adversarial examples are perceptually realistic and more difficult to defend
against with existing defense systems. This potentially provides a new direction
in adversarial example generation and the design of corresponding defenses. We
visualize the spatial transformation based perturbation for different examples and
show that our technique can produce realistic adversarial examples with smooth
image deformation. Finally, we visualize the attention of deep networks with different types of adversarial examples to better understand how these examples are
interpreted.

### [Towards Large yet Imperceptible Adversarial Image Perturbations with Perceptual Color Distance](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Towards_Large_Yet_Imperceptible_Adversarial_Image_Perturbations_With_Perceptual_Color_CVPR_2020_paper.pdf)

In this
work, we drop this assumption by pursuing an approach
that exploits human color perception, and more specifically, minimizing perturbation size with respect to perceptual color distance. Our first approach, Perceptual Color
distance C&W (PerC-C&W), extends the widely-used C&W
approach and produces larger RGB perturbations. PerCC&W is able to maintain adversarial strength, while contributing to imperceptibility. Our second approach, Perceptual Color distance Alternating Loss (PerC-AL), achieves
the same outcome, but does so more efficiently by alternating between the classification loss and perceptual color difference when updating perturbations.


## Defense

### [Smooth Adversarial Training](https://arxiv.org/pdf/2006.14536.pdf)

Here we present evidence to challenge
these common beliefs by a careful study about adversarial training. Our key observation is that the widely-used ReLU activation function significantly weakens
adversarial training due to its non-smooth nature. Hence we propose smooth adversarial training (SAT), in which we replace ReLU with its smooth approximations
to strengthen adversarial training. The purpose of smooth activation functions in
SAT is to allow it to find harder adversarial examples and compute better gradient
updates during adversarial training.


### [Can Adversarial Network Attack be Defended?](https://arxiv.org/pdf/1903.05994.pdf)
- The possiblity of defense aginst adversarial attack on netowk
- propose defense strategies for GNNs against attacks
	1) novel adversarial training strategies
	2) investigate the robustness properties for GNNs granted by the use of smooth defese
	3) two special smooth defense strategies: (smoothing gradient of GNNs)
		a) smoothing distillation
		b) smoothing cross-entropy loss function.
- This work focus on node classification emplying the graph convolutional network. --> what's graph convolutional network.

### [Smoothing Adversarial Training for GNN](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9305289)


## Measurement

### [Exploiting Human Perception for Adversarial Attacks](https://escholarship.org/content/qt2f85f2j6/qt2f85f2j6.pdf)


## Frenquancy

### [Low frequency adversarial perturbation](https://arxiv.org/pdf/1809.08758.pdf)

- black-box setting; restrict the search for adversarial images to a
low frequency domain; n circumvent image transformation defenses even
when both the model and the defense strategy
are unknown.
- motivation: The inherent query inefficiency of gradient estimation and
decision-based attacks stems from the need to search over
or randomly sample from the high-dimensional image
space. Thus, their query complexity depends on the relative adversarial subspace dimensionality compared to the
full image space. One way to improve these methods is to
find a low-dimensional subspace that contains a high density of adversarial examples, which enables more efficient
sampling of useful attack directions.
- Discrete cosine transform

##### [Simple Black-box Adversarial Attacks](Simple Black-box Adversarial Attacks)

- With only the mild assumption of
continuous-valued confidence scores, our highly
query-efficient algorithm utilizes the following
simple iterative principle: we randomly sample a
vector from a predefined orthonormal basis and
either add or subtract it to the target image. Despite its simplicity, the proposed method can be
used for both untargeted and targeted attacks –
resulting in previously unprecedented query efficiency in both settings.


### [Frequency-Tuned Universal Adversarial Attacks](https://arxiv.org/abs/2003.05549)

- we propose to adopt JND thresholds to guide the perceivability of universal adversarial perturbations. Based on this, we propose a frequency-tuned universal attack method to compute universal perturbations and show that our method can realize a good balance between perceivability and effectiveness in terms of fooling rate by adapting the perturbations to the local frequency content. Compared with existing universal adversarial attack techniques, our frequency-tuned attack method can achieve cutting-edge quantitative results. 

- Discrete Cosine Transform; Perception-based JND Thresholds; Frequency-Tuned Universal Perturbations.

