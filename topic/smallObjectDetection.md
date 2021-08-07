### [Augmentation for small object detection](https://arxiv.org/pdf/1902.07296.pdf)

We analyze the current state-of-the-art model, Mask-RCNN, on a challenging dataset, MS COCO. We show that the overlap between small
ground-truth objects and the predicted anchors is much lower than the
expected IoU threshold. We conjecture this is due to two factors; (1) only
a few images are containing small objects, and (2) small objects do not
appear enough even within each image containing them. We thus propose
to oversample those images with small objects and augment each of those
images by copy-pasting small objects many times. 

### [Small Object Detection using Context and Attention](https://arxiv.org/pdf/1912.06319.pdf)

The proposed method uses additional features from different layers as context by concatenating multi-scale features. We also propose object detection with attention mechanism which can focus on the object in image, and it can include contextual information
from target layer.
