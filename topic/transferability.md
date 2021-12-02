### [Data-Free Adversarial Perturbations for Practical Black-Box Attack](https://arxiv.org/abs/2003.01295)

![1](figures/fig1.png)
![1](figures/DFAA.png)
![1](figures/oploss.png)

- do not use the distribution of the training data (universal adversarial perturbation methods)
- learn adversarial perturbation that disturb the internal representation
- maximizeds the divergence between clean images and their adversarial examples in the representation space
- suragation model VS target model --> pre-trained model and fine-tuned model

**comments**
- the transferability is tested in fine-tuned model, and they use the gradient of pre-trained model (should have similar architecure), which means they actually already know some information
- they claim they disturb the internal representation, but their loss function is using the output at logits layer, it is a kind of exaggeration.
- they maximized the divergence equation, but why not other formulation of divergence, why this works better? No ideas.


## Stablizing Gradients

### [Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081)

- MI-FGSM
- They believe that momentum term provides more stable directions and results in more transferable adversarial examples.
- becuase some ppor local maxima in optimization process are only related to specific model. They exist around a data point due to highly non-linear structure of DNN. And hard to transform to other models.
- one-step gradients based method generate more transferable adversarial examples but usually have a low success rate

### [Patch-wise Attack for Fooling Deep Neural Network](https://arxiv.org/abs/2007.06765)

- [patch-wise iterative algorithm](https://github.com/qilong-zhang/Patch-wise-iterative-attack)
- amplification factor to the step size in each iteration
- one pixel's overall gradient overflowing the epsilon=constraint is properly assigned to its surrounding regions by a project kernel
- regionally homogeneous perturbations are strong in attacking defense models
- Patch Map
![1](figures/patchM.png)
![1](figures/pi-fgsm.png)
Wp is a special uniform project kernel, it is to reuse the cut noise to alleviate the disadvantages of direct clipping, increasing the aggregation degree of noise patches

**comments**
- not sure how much the Wp helps... 
- It makes sence to make a patch-wise version but how it is not fair to compare fgsm-based methods, how it compare to patch-based method like square attack...

### [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)

- [Variance Tuning Attack](https://github.com/JHL-HUST/VT)
- instead of directly using the current gradient for the momentum accumulation, they further consider the gradient variance of the previous iteration to tune the current gradient so as to stabilize the update direction and escape from poor local optima.
![1](figures/vmi-fgsm.png)
![1](figures/gradientV.png)

**comments**
- make since to use the gradient variance to make it more stable but why L1 norm???
