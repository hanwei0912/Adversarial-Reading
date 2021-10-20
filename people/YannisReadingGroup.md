### [CrossCLR: Cross-modal Contrastive Learning For Multi-modal Video Representations](https://arxiv.org/pdf/2109.14910.pdf)

- contrastive learning: positive and negtive samples relative to an
ankor point, which yields a flexible principle: pull together
an anchor and a positive sample in the embedding space,
and push apart the anchor from many negative samples.
- dawnback: The cross-modal
loss only ensures that the features from the two modalities map to proximate points in the joint embedding, but
they lack an explicit measure that also ensures that similar
features from the same modality stay close-by in the joint
embedding; The focus of previous cross modal contrastive losses is on the definition of positive
pairs, whereas negative samples are randomly drawn from
the entire distribution
- put similar samples together not only for the data from different modality but also for the same modality
- considering the negtive pairs may share some similarity and ignore then from negtive pairs when calculate the loss

comment: Since the experimental part only do the fine-turning, it might totally failed if the model is trained from scratch.
