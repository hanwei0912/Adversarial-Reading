## Methods

### [Has: Hide-and-seek: Forcing a network to be meticulous for weakly-supervised object and action localization](https://openaccess.thecvf.com/content_ICCV_2017/papers/Singh_Hide-And-Seek_Forcing_a_ICCV_2017_paper.pdf)
- ICCV 2017
- The key idea is to hide patches from an image during training so that the model needs to seek the relevant object parts from what remains

### [ACoL: Adversarial complementary learning for weakly supervised object localization](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.pdf)
- CVPR 2018
- we leverage one classification branch to dynamically localize some discriminative object regions during the forward pass. Although it is usually
responsive to sparse parts of the target objects, this classifier can drive the counterpart classifier to discover new
and complementary object regions by erasing its discovered
regions from the feature maps. With such an adversarial
learning, the two parallel-classifiers are forced to leverage
complementary object regions for classification and can finally generate integral object localization together

### [ADL: Shallow feature matters for weakly supervised object localization](https://openaccess.thecvf.com/content/CVPR2021/papers/Wei_Shallow_Feature_Matters_for_Weakly_Supervised_Object_Localization_CVPR_2021_paper.pdf)
- CVPR 2021
- we propose a simple but
effective Shallow feature-aware Pseudo supervised Object
Localization (SPOL) model for accurate WSOL, which
makes the utmost of low-level features embedded in shallow layers. In practice, our SPOL model first generates the
CAMs through a novel element-wise multiplication of shallow and deep feature maps, which filters the background
noise and generates sharper boundaries robustly. Besides,
we further propose a general class-agnostic segmentation
model to achieve the accurate object mask, by only using
the initial CAMs as the pseudo label without any extra annotation

### [In-sample contrastive learning and consistent attention for weakly supervised object localization](https://openaccess.thecvf.com/content/ACCV2020/papers/Ki_In-sample_Contrastive_Learning_and_Consistent_Attention_for_Weakly_Supervised_Object_ACCV_2020_paper.pdf)
- ACCV 2020
- In this paper, we consider the background as
an important cue that guides the feature activation to cover the sophisticated object region and propose contrastive attention loss. The loss
promotes similarity between foreground and its dropped version, and,
dissimilarity between the dropped version and background. Furthermore,
we propose foreground consistency loss that penalizes earlier layers producing noisy attention regarding the later layer as a reference to provide
them with a sense of backgroundness. 

### [EIL: Erasing integrated learning: A simple yet effective approach for weakly supervised object localization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mai_Erasing_Integrated_Learning_A_Simple_Yet_Effective_Approach_for_Weakly_CVPR_2020_paper.pdf)
- CVPR 2020
- we propose a simple yet powerful approach
by introducing a novel adversarial erasing technique, erasing integrated learning (EIL). By integrating discriminative
region mining and adversarial erasing in a single forwardbackward propagation in a vanilla CNN, the proposed EIL
explores the high response class-specific area and the less
discriminative region simultaneously, thus could maintain
high performance in classification and jointly discover the
full extent of the object

### [SPG: Self-produced guidance for weaklysupervised object localization](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaolin_Zhang_Self-produced_Guidance_for_ECCV_2018_paper.pdf)
- ECCV 2018
- We propose to generate Self-produced
Guidance (SPG) masks which separate the foreground i.e. , the object
of interest, from the background to provide the classification networks
with spatial correlation information of pixels. A stagewise approach is
proposed to incorporate high confident object regions to learn the SPG
masks. The high confident regions within attention maps are utilized
to progressively learn the SPG masks. The masks are then used as an
auxiliary pixel-level supervision to facilitate the training of classification
networks. 

### [I2C: Inter-image communication for weakly supervised localization](https://arxiv.org/pdf/2008.05096.pdf)
- ECCV 2020
- we propose to leverage
pixel-level similarities across different objects for learning more accurate object locations in a complementary way. Particularly, two kinds
of constraints are proposed to prompt the consistency of object features
within the same categories. The first constraint is to learn the stochastic
feature consistency among discriminative pixels that are randomly sampled from different images within a batch. The discriminative information embedded in one image can be leveraged to benefit its counterpart
with inter-image communication. The second constraint is to learn the
global consistency of object features throughout the entire dataset. We
learn a feature center for each category and realize the global feature
consistency by forcing the object features to approach class-specific centers. The global centers are actively updated with the training process.
The two constraints can benefit each other to learn consistent pixel-level
features within the same categories, and finally improve the quality of
localization maps.


### [Unveiling the potential of structure preserving for weakly supervised object localization](https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_Unveiling_the_Potential_of_Structure_Preserving_for_Weakly_Supervised_Object_CVPR_2021_paper.pdf)
- CVPR 2021
- we propose a two-stage
approach, termed structure-preserving activation (SPA), toward fully leveraging the structure information incorporated
in convolutional features for WSOL. First, a restricted activation module (RAM) is designed to alleviate the structuremissing issue caused by the classification network on the basis of the observation that the unbounded classification map
and global average pooling layer drive the network to focus
only on object parts. Second, we designed a post-process
approach, termed self-correlation map generating (SCG)
module to obtain structure-preserving localization maps
on the basis of the activation maps acquired from the first
stage. Specifically, we utilize the high-order self-correlation
(HSC) to extract the inherent structural information retained
in the learned model and then aggregate HSC of multiple
points for precise object localization

### [Keep calm and improve visual feature attribution](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Keep_CALM_and_Improve_Visual_Feature_Attribution_ICCV_2021_paper.pdf)
- ICCV 2021
- we improve CAM by explicitly incorporating a latent variable encoding the location of the cue for recognition in the formulation, thereby subsuming the attribution
map into the training computational graph. The resulting model, class activation latent mapping, or CALM, is
trained with the expectation-maximization algorithm

### [Cutmix: Regularization strategy to train strong classifiers with localizable features](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf)
- ICCV 2019
- patches are cut and pasted among training images where the ground truth labels are also mixed
proportionally to the area of the patches. By making efficient use of training pixels and retaining the regularization effect of regional dropout, CutMix consistently outperforms the state-of-the-art augmentation strategies on CIFAR and ImageNet classification tasks, as well as on the ImageNet weakly-supervised localization task. Moreover, unlike previous augmentation methods, our CutMix-trained
ImageNet classifier, when used as a pretrained model, results in consistent performance gains in Pascal detection
and MS-COCO image captioning benchmarks

### [Danet: Divergent activation for weakly supervised object localization](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xue_DANet_Divergent_Activation_for_Weakly_Supervised_Object_Localization_ICCV_2019_paper.pdf)
- ICCV 2019
- we propose a divergent activation (DA) approach,
and target at learning complementary and discriminative
visual patterns for image classification and weakly supervised object localization from the perspective of discrepancy. To this end, we design hierarchical divergent activation (HDA), which leverages the semantic discrepancy to
spread feature activation, implicitly. We also propose discrepant divergent activation (DDA), which pursues object
extent by learning mutually exclusive visual patterns, explicitly. Deep networks implemented with HDA and DDA,
referred to as DANets, diverge and fuse discrepant yet discriminative features for image classification and object localization in an end-to-end manner


### [PaS: Rethinking class activation mapping for weakly supervised object localization](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600613.pdf)
- ECCV 2020
- CAM approach suffers
from three fundamental issues: (i) the bias of GAP that assigns a higher
weight to a channel with a small activation area, (ii) negatively weighted
activations inside the object regions and (iii) instability from the use
of the maximum value of a class activation map as a thresholding reference.
- We propose three simple
but robust techniques that alleviate the problems, including thresholded
average pooling, negative weight clamping, and percentile as a standard
for thresholding.

### [IVR: Normalization matters in weakly supervised object localization](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Normalization_Matters_in_Weakly_Supervised_Object_Localization_ICCV_2021_paper.pdf)
- ICCV 2021
- In spite
of many WSOL methods proposing novel strategies, there
has not been any de facto standard about how to normalize the class activation map (CAM). Consequently, many
WSOL methods have failed to fully exploit their own capacity because of the misuse of a normalization method. In this
paper, we review many existing normalization methods and
point out that they should be used according to the property
of the given dataset. Additionally, we propose a new normalization method which substantially enhances the performance of any CAM-based WSOL methods. Using the proposed normalization method, we provide a comprehensive
evaluation over three datasets (CUB, ImageNet and OpenImages) on three different architectures and observe significant performance gains over the conventional min-max normalization method in all the evaluated cases

### [GC-Net: Geometry constrained weakly supervised object localization](https://arxiv.org/pdf/2007.09727.pdf)
- 2020
We propose a geometry constrained network, termed GCNet, for weakly supervised object localization (WSOL). GC-Net consists
of three modules: a detector, a generator and a classifier. The detector
predicts the object location defined by a set of coefficients describing a
geometric shape (i.e. ellipse or rectangle), which is geometrically constrained by the mask produced by the generator. The classifier takes the
resulting masked images as input and performs two complementary classification tasks for the object and background. To make the mask more
compact and more complete, we propose a novel multi-task loss function
that takes into account area of the geometric shape, the categorical crossentropy and the negative entropy. In contrast to previous approaches,
GC-Net is trained end-to-end and predict object location without any
post-processing (e.g. thresholding) that may require additional tuning

FAM: Foreground activation maps for weakly supervised object localization

ORNet: Online refinement of low-level feature based activation map for weakly supervised object localization

PSOL: Rethinking the route towards weakly supervised object localization

SLT-Net: Strengthen learning tolerance for weakly supervised object localization

SPOL: Shallow feature matters for weakly supervised object localization

### [Bridging the Gap between Classification and Localization for Weakly Supervised Object Localization](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Bridging_the_Gap_Between_Classification_and_Localization_for_Weakly_Supervised_CVPR_2022_paper.pdf)

### [EVALUATING WEAKLY SUPERVISED OBJECT LOCALIZATION METHODS RIGHT? A STUDY ON HEATMAPBASED XAI AND NEURAL BACKED DECISION TREE](https://openreview.net/pdf?id=X55dLasnEcC)
- ICLR 2023 under review
- They addressed the ill-posed nature of
the problem and showed that WSOL has not significantly improved beyond the
baseline method class activation mapping (CAM).  (1) we perform WSOL
using heatmap-based eXplanaible AI (XAI) methods (2) our model is not class
agnostic since we are interested in the XAI aspect as well. Under similar protocol,
we find that XAI methods perform WSOL with very sub-standard MaxBoxAcc
scores. The experiment is then repeated for the same model trained with Neural
Backed Decision Tree (NBDT) and we found that vanilla CAM yields significantly
better WSOL performance after NBDT training.

### [StarNet: towards Weakly Supervised Few-Shot Object Detection](https://ojs.aaai.org/index.php/AAAI/article/download/16268/16075)
- AAAI 2021
- StarNet - a few-shot model featuring an end-to-end differentiable non-parametric star-model detec- tion and classification head. Through this head, the backbone is meta-trained using only image-level labels to produce good features for jointly localizing and classifying previously un- seen categories of few-shot test tasks using a star-model that geometrically matches between the query and support images (to find corresponding object instances). Being a few-shot de- tector, StarNet does not require any bounding box annota- tions, neither during pre-training, nor for novel classes adap- tation. It can thus be applied to the previously unexplored and challenging task of Weakly Supervised Few-Shot Object Detection (WS-FSOD), where it attains significant improve- ments over the baselines. In addition, StarNet shows signifi- cant gains on few-shot classification benchmarks that are less cropped around the objects (where object localization is key).

## Evaluation
Evaluation for weakly supervised object localization: Protocol, metrics, and datasets

Evaluating weakly supervised object localization methods right
