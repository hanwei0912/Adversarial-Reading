- [Cohen J, Rosenfeld E, Kolter Z (2019) Certified adversarial robustness via randomized smoothing. In: Chaudhuri K, Salakhutdinov R (eds) Proceedings of the 36th International Conference on Machine Learning]

- [Salman H, Yang G, Li J, Zhang P, Zhang H, Razenshteyn I, Bubeck S (2019) Provably robust deep learning via adversarially trained smoothed classifiers. arXiv preprint arXiv:190604584]

- [Kumar A, Levine A, Goldstein T, Feizi S (2020) Curse of dimensionality on randomized smoothing for certifiable robustness. arXiv preprint arXiv:200203239]

- [Protecting JPEG Images Against Adversarial Attacks]



### [Protecting JPEG Images Against Adversarial Attacks](https://arxiv.org/pdf/1803.00940.pdf)

- adaptive JPEG encoder which defends against attacks
	1) high visual quality
	modest increase in encoding time,


### [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf)

[code](https://github.com/locuslab/smoothing)

1. train a network with guassian data augmentation
2. Pa: lower bound on the probability
   Pb: upper bound on the probability
3. Monte Carlo algorithms to compute the probabilities with which f classifies N(x,theta * I) as each class
4. 

### [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/pdf/1906.04584.pdf)

[code](https://github.com/Hadisalman/smoothing-adversarial)

- outperforms all existing provably l2-robust classifiers by a significant margin on imageNet and cifar-10


### [Curse of Dimensionality on Randomized Smoothing for Certifiable Robustness](https://arxiv.org/pdf/2002.03239.pdf)

- extending the smoothing technique to defend against other attack models can be challenging in high-dimensional regime.



