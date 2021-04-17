### [Robustness of saak transform against adversarial attacks](https://ieeexplore.ieee.org/iel7/8791230/8799366/08803240.pdf?casa_token=snoJp85EcsUAAAAA:ZiMzwd3S6Ba24HfdEaF9KUAW9YgZabNuj3bftLR812-ZeSalviMaILfXsW79UMcY4Z7S7CPlpQ)
- abstract: This work investigates the robustness of Saak transform
against adversarial attacks towards high performance image
classification. We develop a complete image classification
system based on multi-stage Saak transform. In the Saak
transform domain, clean and adversarial images demonstrate
different distributions at different spectral dimensions. Selection of the spectral dimensions at every stage can be viewed
as an automatic denoising process. Motivated by this observation, we carefully design strategies of feature extraction, representation and classification that increase adversarial robustness
- Saak transform: Saak transform defines a mapping from three-dimensional
real-valued function (consisting of spatial and spectral dimensions) to a one-dimensional rectified spectral vector. It is presented as a new feature representation method. It consists
of two main ideas: subspace approximation and kernel augmentation. For the former, we build the optimal linear subspace approximation to the original signal space via PCA or
the truncated Karhunen-Loeve Transform (KLT) 

### [Adversarial dual network learning with randomized image transform for restoring attacked images](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8968395)
- abstract: We introduce a randomized nonlinear
transform to disturb and partially destroy the sophisticated pattern of attack noise. We then design a
generative cleaning network to recover the original image content damaged by this nonlinear transform
and remove residual attack noise. We also construct a detector network which serves as the dual network
for the target classifier to be defended, being able to detect patterns of attack noise

### [Generative Cleaning Networks with Quantized Nonlinear Transform for Deep Neural Network Defense](https://openreview.net/pdf?id=SkxOhANKDr)
- abstract: In this paper, we develop a new generative cleaning network with quantized nonlinear transform for
effective defense of deep neural networks. The generative cleaning network,
equipped with a trainable quantized nonlinear transform block, is able to destroy
the sophisticated noise pattern of adversarial attacks and recover the original image content.

### [Defending against adversarial attacks in deep neural networks](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11006/110061C/Defending-against-adversarial-attacks-in-deep-neural-networks/10.1117/12.2519268.short?SSO=1)
- abstract: The method employs a novel signal processing theory as a defense to adversarial perturbations. The method neither modifies the protected network nor requires knowledge of the process for generating adversarial examples. 

### [Local gradients smoothing: Defense against localized adversarial attacks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8658401&casa_token=ToUDz-lu-a0AAAAA:wmjpUQnPigxU5qmefAXZR5Oa8h8k02JiBR00qLv5ox89LXrlk-FGA1TSuc6isSrTvlYNqjK1RA)
- abstract: Driven by the observation
that such attacks introduce concentrated high-frequency
changes at a particular image location, we have developed
an effective method to estimate noise location in gradient
domain and transform those high activation regions caused
by adversarial noise in image domain while having minimal effect on the salient object that is important for correct
classification. Our proposed Local Gradients Smoothing
(LGS) scheme achieves this by regularizing gradients in the
estimated noisy region before feeding the image to DNN for
inference.


## [Defending against adversarial images using basis functions transformations](https://arxiv.org/abs/1803.10840)
- abstract: We study the effectiveness of various approaches that defend against adversarial attacks on
deep networks via manipulations based on basis function representations of images. Specifically, we experiment with low-pass filtering, PCA, JPEG compression, low resolution wavelet
approximation, and soft-thresholding. We evaluate these defense techniques using three types
of popular attacks in black, gray and white-box settings
![daf](figures/defendAbasicfunction.png)
![daf](figures/defendAbasicfunction1.png)
![daf](figures/defendAbasicfunction2.png)

## [Encryption inspired adversarial defense for visual classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9190904&casa_token=Z1Iyo_hyV_QAAAAA:0UdK0p2dbvxrtukVhPI81PLJ6YuWkTOAQEWZ7Lu30-PjYfBU8Zn239IxIBVx-eVODokbHEAI7w)
- abstract: In this paper, we
propose a new adversarial defense which is a defensive transform for both training and test images inspired by perceptual
image encryption methods. The proposed method utilizes a
block-wise pixel shuffling method with a secret key.

![enc](figures/encryption.png)

## [Barrage of Random Transforms for Adversarially Robust Defense](https://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf)
- abstract： In this paper, we explore the idea of stochastically combining a large number of individually weak defenses into
a single barrage of randomized transformations to build
a strong defense against adversarial attacks. We show
that, even after accounting for obfuscated gradients, the
Barrage of Random Transforms (BaRT) is a resilient
defense against even the most difficult attacks, such as
PGD.

![enc](figures/barrage.png)

## [Structure-Preserving Progressive Low-rank Image Completion for Defending Adversarial Attacks](https://arxiv.org/pdf/2103.02781.pdf)
- abstract：In this work, we propose to develop a structure-preserving
progressive low-rank image completion (SPLIC) method to remove unneeded texture details from the
input images and shift the bias of deep neural networks towards global object structures and semantic cues. We formulate the problem into a low-rank matrix completion problem with progressively
smoothed rank functions to avoid local minimums during the optimization process

![str](figures/structure.png)

## [Diminishing the Effect of Adversarial Perturbations via Refining Feature Representation](https://arxiv.org/pdf/1907.01023.pdf)
- abstract:  In this work, we analytically
investigate each layer’s representation of non-perturbed and perturbed images and
show the effect of perturbations on each of these representations. Accordingly, a
method based on whitening coloring transform is proposed in order to diminish the
misrepresentation of any desirable layer caused by adversaries. Our method can be
applied to any layer of any arbitrary model without the need of any modification or
additional training. 

![dim](figures/dimining.png)
![dim](figures/dimining1.png)
![dim](figures/dimining2.png)

## [Ensemble Generative Cleaning With Feedback Loops for Defending Adversarial Attacks](https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Ensemble_Generative_Cleaning_With_Feedback_Loops_for_Defending_Adversarial_Attacks_CVPR_2020_paper.html)
- abstract: In this paper, we develop a new method called ensemble generative cleaning with feedback loops (EGC-FL) for effective defense of deep neural networks. The proposed EGC-FL method is based on two central ideas. First, we introduce a transformed deadzone layer into the defense network, which consists of an orthonormal transform and a deadzone-based activation function, to destroy the sophisticated noise pattern of adversarial attacks. Second, by constructing a generative cleaning network with a feedback loop, we are able to generate an ensemble of diverse estimations of the original clean image. We then learn a network to fuse this set of diverse estimations together to restore the original image. 
![ens](figures/ensemble.png)
![ens](figures/ensemble1.png)

## [Ensemble of Models Trained by Key-based Transformed Images for Adversarially Robust Defense Against Black-box Attacks](https://arxiv.org/abs/2011.07697)
- abstract: We propose a voting ensemble of models trained by using block-wise transformed images with secret keys for an adversarially robust defense. Key-based adversarial defenses were demonstrated to outperform state-of-the-art defenses against gradient-based (white-box) attacks. However, the key-based defenses are not effective enough against gradient-free (black-box) attacks without requiring any secret keys. Accordingly, we aim to enhance robustness against black-box attacks by using a voting ensemble of models. In the proposed ensemble, a number of models are trained by using images transformed with different keys and block sizes, and then a voting ensemble is applied to the models

![key](figures/keybased.png)
