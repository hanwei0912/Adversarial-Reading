## Adversarial attack

### [Frequency Domain Model Augmentation for Adversarial Attack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/974_ECCV_2022_paper.php)
- Spectrum saliency map
- [code](https://github.com/yuyang-long/SSA)
1. All of existing model augmentation methods investigate relationships of different models in spatial domain, which may overlook the essential differences between them.
2. To better uncover the differences among models, we introduce the spectrum saliency map (see Sec. 3.2) from a frequency domain perspective since representation of images in this domain have a fixed pattern, e.g., low-frequency components of an image correspond to its contour.
3. As illustrated in Figure 1 (d~g), spectrum saliency maps (See Sec. 3.2) of different models significantly vary from each other, which clearly reveals that each model has different interests in the same frequency component.



## Weakly Supervised Object Localization

### [Object Discovery via Contrastive Learning for Weakly Supervised Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5458_ECCV_2022_paper.php)
- we propose a novel multiple instance labeling method called object discovery. We further introduce a new contrastive loss under weak supervision where no instance-level information is available for sampling, called weakly supervised contrastive loss (WSCL). WSCL aims to construct a credible similarity threshold for object discovery by leveraging consistent features for embedding vectors in the same class.
- [code](https://github.com/jinhseo/OD-WSCL)
