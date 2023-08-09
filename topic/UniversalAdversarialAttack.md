### [Learning Universal Adversarial Perturbations with Generative Models](https://arxiv.org/pdf/1708.05207.pdf)

### [Art of singular vectors and universal adversarial perturbations](https://openaccess.thecvf.com/content_cvpr_2018/papers/Khrulkov_Art_of_Singular_CVPR_2018_paper.pdf)
- CVPR 2018
- Our approach is based on computing the socalled (p, q)-singular vectors of the Jacobian matrices of
hidden layers of a network.

### [Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations](https://arxiv.org/pdf/1801.08092.pdf)
Existing methods to craft universal perturbations are (i)
task specific, (ii) require samples from the training data distribution, and (iii) perform complex optimizations. Additionally, because of the
data dependence, fooling ability of the crafted perturbations is proportional to the available training data. In this paper, we present a
novel, generalizable and data-free approaches for crafting universal adversarial perturbations. Independent of the underlying task, our
objective achieves fooling via corrupting the extracted features at multiple layers. Therefore, the proposed objective is generalizable to
craft image-agnostic perturbations across multiple vision tasks such as object recognition, semantic segmentation, and depth
estimation

### [Fast Feature Fool: A data independent approach to universal adversarial perturbations](https://arxiv.org/pdf/1707.05572.pdf)
It is also observed that these perturbations generalize across multiple networks trained on
the same target data. However, these algorithms require training data on which the CNNs
were trained and compute adversarial perturbations via complex optimization. The fooling performance of these approaches is directly proportional to the amount of available
training data. This makes them unsuitable for practical attacks since its unreasonable
for an attacker to have access to the training data. In this paper, for the first time, we
propose a novel data independent approach to generate image agnostic perturbations for
a range of CNNs trained for object recognition. We further show that these perturbations
are transferable across multiple network architectures trained either on same or different data. In the absence of data, our method generates universal perturbations efficiently
via fooling the features learned at multiple layers thereby causing CNNs to misclassify.
Experiments demonstrate impressive fooling rates and surprising transferability for the
proposed universal perturbations generated without any training data.

### [Universal Adversarial Perturbation via Prior Driven Uncertainty Approximation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Universal_Adversarial_Perturbation_via_Prior_Driven_Uncertainty_Approximation_ICCV_2019_paper.pdf)
- ICCV 2019
- Specifically, a Monte Carlo sampling method is deployed to activate more neurons to increase the model uncertainty for a
better adversarial perturbation.

### [Ask, Acquire, and Attack: Data-free UAP Generation using Class Impressions](https://openaccess.thecvf.com/content_ECCV_2018/papers/Konda_Reddy_Mopuri_Ask_Acquire_and_ECCV_2018_paper.pdf)
-ECCV 2018
In this paper, for data-free scenarios, we
propose a novel approach that emulates the effect of data samples with
class impressions in order to craft UAPs using data-driven objectives.
Class impression for a given pair of category and model is a generic
representation (in the input space) of the samples belonging to that category. Further, we present a neural network based generative model that
utilizes the acquired class impressions to learn crafting UAPs

### [NAG: Network for Adversary Generation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mopuri_NAG_Network_for_CVPR_2018_paper.pdf)
- CVPR 2018
- In this paper, we propose
for the first time, a generative approach to model the distribution of adversarial perturbations. The architecture of
the proposed model is inspired from that of GANs and is
trained using fooling and diversity objectives. Our trained
generator network attempts to capture the distribution of
adversarial perturbations for a given classifier and readily
generates a wide variety of such perturbations. 
