## 2024

### [Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification](https://arxiv.org/pdf/2411.05698?)

### [On the Concept Trustworthiness in Concept Bottleneck Models](https://arxiv.org/abs/2403.14349)
- concept trustworthness score  **We need to check it**
- three modules: cross-layer alignment, prediction alignment, cross-image alignment
https://github.com/hqhQAQ/ProtoCBM
CUB: pcbm 58.8 this 68.3
 (First step) To calculate this evaluation
metric, our work first generates the corresponding region
of each concept on the input image (i.e., the image region
that the concept is predicted from). (Second step) Next, this
evaluation metric estimates the concept trustworthiness according to whether the corresponding region of the concept
is consistent with the ground-truth

### [Sparse Concept Bottleneck Models: Gumbel Tricks in Contrastive Learning](https://arxiv.org/pdf/2404.03323)
- concept matrix search algorithm: define image-concept matrix, and class-concept matrix
- contrastive-CBM
- sparse-CBM （Gumbel-max trick） **We need to check it**
https://github.com/Andron00e/SparseCBM
CUB: pcbm 63.92 this 80.02

### [Interpreting Pretrained Language Models via Concept Bottlenecks](https://arxiv.org/pdf/2311.05014)
- Concept-Bottleneck-Enabled Pretrained Language Models
- learn from noisy concept label
- concept-level mixup

### [Can we Constrain Concept Bottleneck Models to Learn Semantically Meaningful Input Features?](https://arxiv.org/pdf/2402.00912)
- introduce a new synthetic image dataset with fine-grained concept annotation **we need to check**
- perform an in-depth analysis of CBM and conclude two factors are critical for CBMs to learn semantically meaningful input features: accuracy of concept annotations and high variability in the combinations of concepts co-occurring

-no code yet

### [AnyCBMs: How to Turn Any Black Box into a Concept Bottleneck Model](https://arxiv.org/pdf/2405.16508)
- **we might need to use this**
- necessitate training a new model from the beginning, consuming significant resources
and failing to utilize already trained large models. To address this issue, we introduce “AnyCBM”, a
method that transforms any existing trained model into a Concept Bottleneck Model with minimal
impact on computational resources

### [Driving through the Concept Gridlock: Unraveling Explainability Bottlenecks in Automated Driving](https://openaccess.thecvf.com/content/WACV2024/papers/Echterhoff_Driving_Through_the_Concept_Gridlock_Unraveling_Explainability_Bottlenecks_in_Automated_WACV_2024_paper.pdf)

### [Interpretable Concept Bottlenecks to Align Reinforcement Learning Agents](https://arxiv.org/pdf/2401.05821)

### [The relational bottleneck as an inductive bias for efficient abstraction](https://www.sciencedirect.com/science/article/pii/S1364661324000809)

### [A Survey on Information Bottleneck](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10438074)

### [Auxiliary Losses for Learning Generalizable Concept-based Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/555479a201da27c97aaeed842d16ca49-Paper-Conference.pdf)

### [Learning to Receive Help:Intervention-Aware Concept Embedding Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/770cabd044c4eacb6dc5924d9a686dce-Paper-Conference.pdf)

### [BREAKING THE ATTENTION BOTTLENECK](https://arxiv.org/pdf/2406.10906)

### [Incremental Residual Concept Bottleneck Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Shang_Incremental_Residual_Concept_Bottleneck_Models_CVPR_2024_paper.pdf)

### [Beyond Concept Bottleneck Models:How to Make Black Boxes Intervenable?](https://arxiv.org/pdf/2401.13544)

### [ENERGY-BASED CONCEPT BOTTLENECK MODELS: UNIFYING PREDICTION, CONCEPT INTERVENTION, AND PROBABILISTIC INTERPRETATIONS](https://arxiv.org/pdf/2401.14142)
https://github.com/xmed-lab/ECBM
CUB: pcbm 63.5 this 81.2

### [Stochastic Concept Bottleneck Models](https://arxiv.org/pdf/2406.19272)
We propose Stochastic Concept Bottleneck Models (SCBMs),
a novel approach that models concept dependencies. In SCBMs, a single-concept
intervention affects all correlated concepts, thereby improving intervention effectiveness. Unlike previous approaches that model the concept relations via an
autoregressive structure, we introduce an explicit, distributional parameterization
that allows SCBMs to retain the CBMs’ efficient training and inference procedure.
Additionally, we leverage the parameterization to derive an effective intervention
strategy based on the confidence region. 

### [Explain via Any Concept: Concept Bottleneck Model with Open Vocabulary Concepts](https://arxiv.org/pdf/2408.02265)
 (1) Aligning the feature space of a trainable image feature
extractor with that of a CLIP’s image encoder via a prototype based
feature alignment; (2) Simultaneously training an image classifier on the
downstream dataset; (3) Reconstructing the trained classification head
via any set of user-desired textual concepts encoded by CLIP’s text encoder. To reveal potentially missing concepts from users, we further propose to iteratively find the closest concept embedding to the residual
parameters during the reconstruction until the residual is small enough.
To the best of our knowledge, our “OpenCBM” is the first CBM with
concepts of open vocabularies, providing users the unique benefit such
as removing, adding, or replacing any desired concept to explain the
model’s prediction even after a model is trained.

- obtaining the importance of any desired concept
- the discovery of missing concepts
- concept removal from an unknown concept set
- concept adding, removal and replacement in a known concept set

### [Improving Concept Alignment in Vision-Language Concept Bottleneck Models](https://arxiv.org/pdf/2405.01825)
 it is desired to build CBMs with concepts defined by human
experts rather than LLM-generated ones to make them
more trustworthy. In this work, we closely examine the
faithfulness of VLM concept scores for such expert-defined
concepts in domains like fine-grained bird species and animal
classification. Our investigations reveal that VLMs like CLIP
often struggle to correctly associate a concept with the
corresponding visual input, despite achieving a high classification performance. This misalignment renders the resulting
models difficult to interpret and less reliable. To address
this issue, we propose a novel Contrastive Semi-Supervised
(CSS) learning method that leverages a few labeled concept
samples to activate truthful visual concepts and improve
concept alignment in the CLIP model.
- they addressed the locality faithfulness as limitations


### [CONCEPT BOTTLENECK GENERATIVE MODELS](https://openreview.net/pdf?id=L9U5MJJleF)
The concept bottleneck layer partitions the generative model into three
parts: the pre-concept bottleneck portion, the CB layer, and the post-concept bottleneck portion. To train CB generative models, we complement the traditional
task-based loss function for training generative models with a concept loss and an
orthogonality loss. The CB layer and these loss terms are model agnostic, which
we demonstrate by applying the CB layer to three different families of generative
models: generative adversarial networks, variational autoencoders, and diffusion
models.

## 2023

### [A Closer Look at the Intervention Procedure of Concept Bottleneck Models](https://proceedings.mlr.press/v202/shin23a/shin23a.pdf)
In this work,
we develop various ways of selecting intervening
concepts to improve the intervention effectiveness and conduct an array of in-depth analyses as
to how they evolve under different circumstances.
Specifically, we find that an informed intervention
strategy can reduce the task error more than ten
times compared to the current baseline under the
same amount of intervention counts in realistic
settings, and yet, this can vary quite significantly
when taking into account different intervention
granularity. We verify our findings through comprehensive evaluations, not only on the standard
real datasets, but also on synthetic datasets that we
generate based on a set of different causal graphs.
We further discover some major pitfalls of the current practices which, without a proper addressing,
raise concerns on reliability and fairness of the
intervention procedure.

### [Towards a Deeper Understanding of Concept Bottleneck Models Through End-to-End Explanation](https://arxiv.org/pdf/2302.03578)
- checking the relevance both from the input to the concept vector, confirming the relevance is distributed among the input feature, and from the concept vector to the final prediction.
- quatitivative evaluation to measure the distance between the maximum input feature relevance and the ground truthe location (LRP,IG)
- propotion of relevance as a measurement for explaining concept importance 

### [Interpreting Pretrained Language Models via Concept Bottlenecks](https://arxiv.org/pdf/2311.05014)
- ChatGPT-guided Concept augmentation with concept-level mixup

### [Interactive Concept Bottleneck Models](https://ojs.aaai.org/index.php/AAAI/article/view/25736/25508)
We extend CBMs
to interactive prediction settings where the model can query
a human collaborator for the label to some concepts. We develop an interaction policy that, at prediction time, chooses
which concepts to request a label for so as to maximally improve the final prediction. We demonstrate that a simple policy
combining concept prediction uncertainty and influence of the
concept on the final prediction achieves strong performance
and outperforms static approaches as well as active feature
acquisition methods proposed in the literature
- concept prediction uncertainty
- concept importance sore
- acquisition cost

### [Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Language_in_a_Bottle_Language_Model_Guided_Concept_Bottlenecks_for_CVPR_2023_paper.pdf)
They address these shortcomings and
are first to show how to construct high-performance CBMs
without manual specification of similar accuracy to black
box models. 


### [Learning Bottleneck Concepts in Image Classification](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Learning_Bottleneck_Concepts_in_Image_Classification_CVPR_2023_paper.pdf)
Combine the bottleneck concept learner in self-supervision, (mixture the prototype network with concept, instead of learning prototype, learning concept feature)
they have a feature aggregation to map attention output to concept feature
they have reconstruction loss, contrastive loss; for the concept regularizer, they introduce individual consistency

This paper proposes Bottleneck Concept Learner (BotCL), which represents an image solely by the presence/absence of concepts learned
through training over the target task without explicit supervision over the concepts. It uses self-supervision and tailored regularizers so that learned concepts can be humanunderstandable. 
- concept regularization: individual consistency, min-batch concept vector distance should be different from each other; mutual distinctiveness, the average concept vector from different min-batch should be differrent
- quantization loss: less concept should be selected
Evaluation:
- coverage
- completeness
- purity
- distinctiveness
- concept discovery rate
- concept consistency
- mutual information between concept

### [Do Concept Bottleneck Models Obey Locality?](https://openreview.net/pdf?id=F6RPYDUIZr)
 Recent work, however, strongly suggests
that this assumption may fail to hold in Concept Bottleneck Models (CBMs),
a quintessential family of concept-based interpretable architectures. In this
paper, we investigate whether CBMs correctly capture the degree of conditional
independence across concepts when such concepts are localised both spatially, by
having their values entirely defined by a fixed subset of features, and semantically,
by having their values correlated with only a fixed subset of predefined concepts.
To understand locality, we analyse how changes to features outside of a concept’s
spatial or semantic locality impact concept predictions. Our results suggest that
even in well-defined scenarios where the presence of a concept is localised to a
fixed feature subspace, or whose semantics are correlated to a small subset of
other concepts, CBMs fail to learn this locality

### [LABEL-FREE CONCEPT BOTTLENECK MODELS](https://arxiv.org/pdf/2304.06129)
Label-free CBM is a novel framework to transform any neural network
into an interpretable CBM without labeled concept data, while retaining a high accuracy. Our Label-free CBM has many advantages, it is: scalable we present the first CBM scaled to ImageNet, efficient - creating a CBM takes only a few hours even for very large datasets, and automated - training it for a new dataset requires minimal human effort.

Advantages: scalable, efficient, automated

Using GPT to generate and filter concept set, then learn concept bottleneck layer (sim dis) and sparse final layer

### [SURROCBM: CONCEPT BOTTLENECK SURROGATE MODELS FOR GENERATIVE POST-HOC EXPLANATION](https://arxiv.org/pdf/2310.07698)
They addressing three challenges: 1) briding the gap between concepts for data and post-hoc explanations; 2) aligning the shared related concepts for multiple classifier; 3) ensuring high fidelity of surrogate models.

They use SurroCBM to explain a source black-box model. They propose three loss for training: identifiability loss; fidelity loss; and explainability loss. In explainability loss, there are disentanglement of concept, sparsity of explanation mask, simplicity of decision trees.

### [Probabilistic Concept Bottleneck Models](https://arxiv.org/pdf/2306.01574)

It attempts to provide a reliable interpretation against the ambiguous in the data by proposing ProbCBM and using probabilistic embedding module to learn a probabilistic concept embedding based on the backbone.


### [POST-HOC CONCEPT BOTTLENECK MODELS](https://arxiv.org/pdf/2205.15480)
PRoblem: CBMs are restrictive in practice as they require dense concept
annotations in the training data to learn the bottleneck; CBMs often do not match the accuracy of an unrestricted neural network, reducing the incentive to deploy them in practice

Solution: Post-hoc Concept Bottleneck models (PCBMs) 
can turn any neural network into a PCBM without sacrificing model performance
while still retaining the interpretability benefits. When concept annotations are
not available on the training data, we show that PCBM can transfer concepts from
other datasets or from natural language descriptions of concepts via multimodal
models. A key benefit of PCBM is that it enables users to quickly debug and
update the model to reduce spurious correlations and improve generalization to
new distributions. PCBM allows for global model edits, which can be more
efficient than previous works on local interventions that fix a specific prediction.
Through a model-editing user study, we show that editing PCBMs via conceptlevel feedback can provide significant performance gains without using data from
the target domain or model retraining.

## 2022

### [Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off](https://proceedings.neurips.cc/paper_files/paper/2022/file/867c06823281e506e8059f5c13a57f75-Paper-Conference.pdf)
- concept alignment score: measures how much learnt concept representation can be trusted as faithful representations of their ground truth concept labels

### [Addressing Leakage in Concept Bottleneck Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/944ecf65a46feb578a43abfd5cddd960-Paper-Conference.pdf)

Disparity between hard CBM and soft CBM:
1. Soft concepts rely on this information-rich
representation, which is in contrast with hard concepts, where label predictor only accepts binary
concepts.
2. While soft concepts may improve predictive performance, this improvement comes at a cost.
3. Soft concepts allow the concept predictor to convey unintended information about the label that would otherwise not be available to the label predictor. A flexible predictor, such as a deep neural network, can quickly learn such encoding patterns and improve its prediction using the information leaked by the concept predictor.

Disparity caused by:
1. having an insufficient concept set: a result of the Markovian assumption under which CBMs operate: all the information about the label in the input must be captured by the concepts
2. having an inflexible concept predictor: CBMs predict concepts independently and cannot capture correlations between concepts. These correlations are important since some combinations of concepts may be unlikely or impossible

Solution of the problem:
1. circumvent the Markovian assumption by introducing a side-channel to learn a set of latent concepts alongside the known concepts that are needed for accurate predictions
2. introduce an autoregressive configuration for the concept predictor that significantly improves their flexibility and it allows them to capture
correlations in the concepts

## 2021

### [DO CONCEPT BOTTLENECK MODELS LEARN AS INTENDED?](https://arxiv.org/pdf/2105.04289)

This work reviews if the CBM fulfill the three desiderata, they useIngerated Gradients as a tool to evaluate which input features are relevant to each concept and which concepts are relevant to each target, then they find CBMs do not provide concept interpretability and do not always predict target values based onthe  concept and may not intervenable.

## 2020
### [Concept Bottleneck Models](https://arxiv.org/pdf/2007.04612)
This work were the first to formally define Concept Bottleneck Models (CBMs) as a pair $(f,g)$ consisting of a backbone network $f: \cX \to \real^{C}$, which maps an input image $\vx \in \cX$ to a concept space $\real^{C}$ containing $C$ predefined concepts, and a classifier $g: \real^C \to \real^K$, which maps the predicted concept embedding to one of $K$ target classes. 
The final predicted label is given by $l = \arg \max_i g_i(f(\vx))$, where $g_i(\cdot)$ represents the predicted logit value for the class $i^{th}$. Let $l_{gt}$ denote the ground truth class label of $\vx$ and $c_{gt}$ denotes the ground truth concept label. 
A CBM addresses three desiderata: \emph{Interpretability} \ie being able to note which concepts are important for the targets, \emph{Predictablity}, \ie being able to predict the targets from the concepts alone, and \emph{Intervenability}, \ie being able to replace predicted concept values with ground truth values to improve predictive performance. 
A CBM model should achieve both concept accuracy and class accuracy to perform the classification task while maintaining explainability effectively. Furthermore, This work consider \emph{independent bottleneck}, \emph{sequential bottleneck}, and \emph{joint bottleneck} to learn a concept bottleneck model with different costs.
