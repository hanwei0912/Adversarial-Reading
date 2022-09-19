## Interpretability of NLP Transformers

#### [What does BERT look at? An analysis of BERT’s attention](https://arxiv.org/abs/1906.04341)
- [code](https://github.com/clarkkev/attention-analysis.)
- compute the average entropy of each head’s attention distribution
- measured entropies for all attention heads from only the [CLS] token
- evaluate attention heads on labeled datasets for tasks like dependency parsing

#### [exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models](https://arxiv.org/abs/1910.05276)
- [code](https://github.com/bhoov/exbert)
- interactive tool: provides insights into the meaning of the
contextual representations by matching a human-specified input to similar contexts
in a large annotated dataset. By aggregating the annotations of the matching similar
contexts, EXBERT helps intuitively explain what each attention-head has learned.

#### [InterpreT: An interactive visualization tool for interpreting transformers](https://aclanthology.org/2021.eacl-demos.17/)
- InterpreT: (1)ability to track and visualize token embeddings through each layer of
a Transformer;(2)highlight distances between
certain token embeddings through illustrative
plots (3)identify task-related functions of attention heads by using new metrics
-  its functionalities are demonstrated through the analysis of
model behaviours for two disparate tasks: Aspect Based Sentiment Analysis (ABSA) and
the Winograd Schema Challenge (WSC)

#### [T3-vis: visual analytic for training and fine-tuning transformers in NLP](https://aclanthology.org/2021.emnlp-demo.26/)
- [code](https://github.com/raymondzmc/T3-Vis)
- offers an intuitive overview that allows the user to
explore different facets of the model (e.g., hidden states, attention) through interactive visualization, and allows a suite of built-in algorithms that compute the importance of model
components and different parts of the input
sequence

#### [The language interpretability tool: Extensible, interactive visualizations and analysis for NLP models](https://arxiv.org/abs/2008.05122)
- [code](https://github.com/PAIR-code/lit)
- LIT integrates local explanations, aggregate analysis, and counterfactual generation into a streamlined, browser-based interface to enable rapid exploration and error analysis.

## Interpreting vision transformers

#### [Transformer interpretability beyond attention visualization](https://arxiv.org/abs/2012.09838)
- [code](https://github.com/hila-chefer/Transformer-Explainability)
- LRP based

## Multimodal interpretability

#### [Behind the scene: Revealing the secrets of pre-trained vision-and-language models](https://arxiv.org/abs/2005.07310)
-  (i) Pre-trained models exhibit a propensity for attending
over text rather than images during inference. (ii) There exists a subset of
attention heads that are tailored for capturing cross-modal interactions. (iii) Learned attention matrix in pre-trained models demonstrates patterns coherent with the latent alignment between image regions and textual words. (iv) Plotted attention patterns reveal visually-interpretable
relations among image regions. (
v) Pure linguistic knowledge is also effectively encoded in the attention heads. 

#### [Probing multimodal embeddings for linguistic properties: the visual-semantic case](https://arxiv.org/abs/2102.11115)
- (i) discuss the formalization of probing tasks for embeddings of image-caption pairs, (ii) define three concrete probing tasks within our general framework, (iii) train classifiers to probe for those properties, and (iv) compare various state-of-the-art embeddings under the lens of the proposed probing tasks

#### [Probing imagelanguage transformers for verb understanding](https://aclanthology.org/2021.findings-acl.318.pdf)
-  shedding light on the
quality of their pretrained representations –
in particular, if these models can distinguish
different types of verbs or if they rely solely
on nouns in a given sentence.

#### [Are vision-language transformers learning multimodal representations? A probing perspective](https://hal.archives-ouvertes.fr/hal-03521715/document)
-  compare pre-trained and finetuned representations at a vision, language and multimodal
level.

#### [VL-InterpreT: An Interactive Visualization Tool for Interpreting Vision-Language Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Aflalo_VL-InterpreT_An_Interactive_Visualization_Tool_for_Interpreting_Vision-Language_Transformers_CVPR_2022_paper.pdf)
- [code](https://github.com/IntelLabs/VL-InterpreT)
