#### [AdCo: Adversarial Contrast for Efficient Learning of Unsupervised Representations from Self-Trained Negative Adversaries](https://arxiv.org/abs/2011.08435)

- Constrastive learning: maintain a queue of negative samples over minibatches
- directily learn a set of negative adversaries playing against the self-trained represetntaion
- two players: the represetntation networks vs. negative adversaries

#### [Style-based Point Generator with Adversarial Rendering for Point Cloud Completion](https://arxiv.org/abs/2103.02535)

- Style-based Point Generator with Adversarial Rendering (SpareNet) for point cloud completion


## Attacks

#### [Simulating Unknown Target Models for Query-Efficient Black-box Attacks](https://arxiv.org/abs/2009.00960)

- motivation: in black-box setting, query complexity remains high. 
- This work aims to train a generalized substitute model called "simulator"
- build the training data with the form of multiple tasks by collecting query sequences generated during the attacks of various existing netowks
- learning: mean square error-based knowledge-distillation loss
______
*Query-based Attacks*
	- score-based: output scores; estimated gradient with zeroth-order optimizations
	- decision-based: output label; 
*Transfer-based Attack*
	Generated adversarial examples on a source model and then transfer them to the target model.
*Meta-learning*
	- MetaAdvDet: detect new types of adversarial attacks with high accuracy
	- Meta Attack: trains auto-encoder to predict the gradients of a target model to reduce the query complexity
___________

- Bandits attack algorithm
- n pre-trained classification netowrks
- Inner update: with trainneing data; train simulator network with all the score-input pairs from n pre-trained networks
- Outer updata: with testing data

#### [Understanding the Robustness of Skeleton-based Action Recognition under Adversarial Attack](https://arxiv.org/abs/2103.05347)

- examine the robustness of s-o-t-a action recognizers against adversarial attack
- proposed a new attack to 3D recognizers

#### [IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking](https://arxiv.org/abs/2103.14938)

- decision-based black-box attack method for visual object tracking
- In contrast to existing black-box adversarial attack methods that deal with static images for image classification, we propose IoU attack that sequentially generates perturbations based on the predicted IoU scores from both current and historical frames. By decreasing the IoU scores, the proposed attack method degrades the accuracy of temporal coherent bounding boxes (i.e., object motions) accordingly. In addition, we transfer the learned perturbations to the next few frames to initialize temporal motion attack. We validate the proposed IoU attack on state-of-the-art deep trackers (i.e., detection based, correlation filter based, and long-term trackers). 



## Defense

### Detection

#### [LiBRe: A Practical Bayesian Approach to Adversarial Detection](https://arxiv.org/abs/2103.14835)

- Lightweight Bayesian Refinement: leveraging Bayesian neural networks for adversarial detection
- few-layer deep ensemble variational + pre-training + fine-tuning
- converts last few layers to be Bayesian + reuses the pre-trained parameters
- launches several-round adversarial detection-oriented fine-tuning

### robustness

#### [Adversarial Robustness under Long-Tailed Distribution](https://arxiv.org/abs/2104.02703)

- realistic scenarios; recognition performance & adversarial robustness
- study on existing long-tailed recognition method
	1. natural accuracy is realtively easy to improve
	2. fake gain of robust accuracy exists under unreliable ecaluation
	3. boundary error limits the promotion of invariant classifier

#### [Zero-shot Adversarial Quantization](https://arxiv.org/abs/2103.15263)

- zero-shot model quantization without accesing training data,  a tiny number of quantization methods adopt either post-training quantization or batch normalization statistics-guided data generation for fine-tuning
-  we propose a zero-shot adversarial quantization (ZAQ) framework, facilitating effective discrepancy estimation and knowledge transfer from a full-precision model to its quantized model. This is achieved by a novel two-level discrepancy modeling to drive a generator to synthesize informative and diverse data examples to optimize the quantized model in an adversarial learning fashion. 

[code](https://github.com/FLHonker/ZAQ-code)
This is achieved by a novel two-level discrepancy modeling to drive a generator to synthesize informative and diverse data examples to optimize the quantized model in an adversarial learning fashion.

#### [Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2103.08896)

AdvCAM is an attribution map of an image that is manipulated to increase the classification score. This manipulation is realized in an anti-adversarial manner, which perturbs the images along pixel gradients in the opposite direction from those used in an adversarial attack. It forces regions initially considered not to be discriminative to become involved in subsequent classifications, and produces attribution maps that successively identify more regions of the target object. In addition, we introduce a new regularization procedure that inhibits the incorrect attribution of regions unrelated to the target object and limits the attributions of the regions that already have high scores. 

#### [Robust and Accurate Object Detection via Adversarial Learning](https://arxiv.org/abs/2103.13886)

- Noting that most state-of-the-art object detectors benefit from fine-tuning a pre-trained classifier, we first study how the classifiers' gains from various data augmentations transfer to object detection. The results are discouraging; the gains diminish after fine-tuning in terms of either accuracy or robustness. 
-  This work instead augments the fine-tuning stage for object detectors by exploring adversarial examples, which can be viewed as a model-dependent data augmentation. Our method dynamically selects the stronger adversarial images sourced from a detector's classification and localization branches and evolves with the detector to ensure the augmentation policy stays current and relevant. This model-dependent augmentation generalizes to different object detectors better than AutoAugment, a model-agnostic augmentation policy searched based on one particular detector

#### [Multi-Objective Interpolation Training for Robustness to Label Noise](https://arxiv.org/abs/2012.04462)

- we propose a Multi-Objective Interpolation Training (MOIT) approach that jointly exploits contrastive learning and classification to mutually help each other and boost performance against label noise.
- andard supervised contrastive learning degrades in the presence of label noise and propose an interpolation training strategy to mitigate this behavior. We further propose a novel label noise detection method that exploits the robust feature representations learned via contrastive learning to estimate per-sample soft-labels whose disagreements with the original labels accurately identify noisy samples. 

#### [Limitations of Post-Hoc Feature Alignment for Robustness](https://arxiv.org/abs/2103.05898)

- Feature alignment is an approach to improving robustness to distribution shift that matches the distribution of feature activations between the training distribution and test distribution.

#### [Can audio-visual integration strengthen robustness under multimodal attacks?](https://arxiv.org/abs/2104.02000)

 we propose to make a systematic study on machines multisensory perception under attacks. We use the audio-visual event recognition task against multimodal adversarial attacks as a proxy to investigate the robustness of audio-visual learning. We attack audio, visual, and both modalities to explore whether audio-visual integration still strengthens perception and how different fusion mechanisms affect the robustness of audio-visual models. For interpreting the multimodal interactions under attacks, we learn a weakly-supervised sound source visual localization model to localize sounding regions in videos. To mitigate multimodal attacks, we propose an audio-visual defense approach based on an audio-visual dissimilarity constraint and external feature memory banks.

