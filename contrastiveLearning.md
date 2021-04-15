## Contrastive learning

The basic idea of contrastive learning is that the samples should be close to the positive samples and far away from the negtive samples. To evaluate such kind of simlarity, they propose a formular which is very close to the cross-entropy. The difference is that the logit for the ground true is replaced by a positive sample, while the rest by N negtive samples.

Then the question becomes, how to select the positive sample and negative sampels.


### [Deep InfoMax](https://arxiv.org/abs/1808.06670)

In this paper, they have three features, i.e. the globel feature (aggregated all the feature), positive local feature from the same image, negtive local feature from another sample. They maximize mutual information between local features and global features.

### [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)

It is method for any format media. They try to fond some invariance featrue that is not change very quick. For instance, the different time window for audios/vedios. And the different spacial window of images. They use the global summary vector of the first pass and the local feature vectors of the second pass.

### [MoCo](https://arxiv.org/pdf/1911.05722.pdf)

They us a momentum encoder to generate negative samples, and when backpropogating from the contrastive loss, there is no gradient to the encoder of the negative samples. This momentum encoder maintains the current negative candidates pool with a queue. And the queue is updated with momentum. It coppy the parameter of the encoder of positive sample to negative sample with momentum.

### [SimCLR](https://arxiv.org/abs/2002.05709)

In this paper, they use transformation to reconstruct the negative samples.

