## 2025

### [VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance](https://arxiv.org/pdf/2408.01432)
First, the concepts predicted
by CBL often mismatch the input image, raising doubts about the faithfulness of
interpretation. Second, it has been shown that concept values encode unintended
information: even a set of random concepts could achieve comparable test accuracy
to state-of-the-art CBMs. To address these critical limitations, in this work, we
propose a novel framework called Vision-Language-Guided Concept Bottleneck
Model (VLG-CBM) to enable faithful interpretability with the benefits of boosted
performance. Our method leverages off-the-shelf open-domain grounded object detectors to provide visually grounded concept annotation, which largely enhances the
faithfulness of concept prediction while further improving the model performance.
In addition, we propose a new metric called Number of Effective Concepts (NEC)
to control the information leakage and provide better interpretability. 

### [Zero-shot Concept Bottleneck Models](https://arxiv.org/pdf/2502.09018)
However, they require target task training to learn input-to-concept
and concept-to-label mappings, incurring target
dataset collections and training resources. In this
paper, we present zero-shot concept bottleneck
models (Z-CBMs), which predict concepts and
labels in a fully zero-shot manner without training neural networks. Z-CBMs utilize a largescale concept bank, which is composed of millions of vocabulary extracted from the web, to
describe arbitrary input in various domains. For
the input-to-concept mapping, we introduce concept retrieval, which dynamically finds inputrelated concepts by the cross-modal search on
the concept bank. In the concept-to-label inference, we apply concept regression to select essential concepts from the retrieved concepts by
sparse linear regression. Through extensive experiments, we confirm that our Z-CBMs provide
interpretable and intervenable concepts without
any additional training.

### [Coarse-to-Fine Concept Bottleneck Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/bdeab378efe6eb289714e2a5abc6ed42-Paper-Conference.pdf)
This work targets ante hoc interpretability, and specifically Concept Bottleneck Models (CBMs). Our goal is to design a framework that admits a highly
interpretable decision making process with respect to human understandable concepts, on two levels of granularity. To this end, we propose a novel two-level
concept discovery formulation leveraging: (i) recent advances in vision-language
models, and (ii) an innovative formulation for coarse-to-fine concept selection via
data-driven and sparsity-inducing Bayesian arguments. Within this framework,
concept information does not solely rely on the similarity between the whole image
and general unstructured concepts; instead, we introduce the notion of concept
hierarchy to uncover and exploit more granular concept information residing in
patch-specific regions of the image scene.

### [The Decoupling Concept Bottleneck Model](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10740789)
This paper proves that insufficient concept information
can lead to an inherent dilemma of concept and label distortions
in CBM. To address this challenge, we propose the Decoupling
Concept Bottleneck Model (DCBM), which comprises two phases:
1) DCBM for prediction and interpretation, which decouples heterogeneous information into explicit and implicit concepts while
maintaining high label and concept accuracy, and 2) DCBM for
human-machine interaction, which automatically corrects labels
and traces wrong concepts via mutual information estimation. The
construction of the interaction system can be formulated as a light
min-max optimization problem. Extensive experiments expose the
success of alleviating concept/label distortions, especially when concepts are insufficient. In particular, we propose the Concept Contribution Score (CCS) to quantify the interpretability of DCBM.
Numerical results demonstrate that CCS can be guaranteed by
the Jensen-Shannon divergence constraint in DCBM. Moreover,
DCBM expresses two effective human-machine interactions, including forward intervention and backward rectification, to further promote concept/label accuracy via interaction with human
experts.

### [Editable Concept Bottleneck Models](https://arxiv.org/pdf/2405.15476)
In many scenarios, we often need to remove/insert some training data or new concepts from trained CBMs
for reasons such as privacy concerns, data mislabelling, spurious concepts, and concept annotation errors. Thus, deriving efficient editable
CBMs without retraining from scratch remains a
challenge, particularly in large-scale applications.
To address these challenges, we propose Editable
Concept Bottleneck Models (ECBMs). Specifically, ECBMs support three different levels of
data removal: concept-label-level, concept-level,
and data-level. ECBMs enjoy mathematically rigorous closed-form approximations derived from
influence functions that obviate the need for retraining. 

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
- concepts preparation
- performance
- editing model

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
We propose a new approach using concept
bottlenecks as visual features for control command predictions and explanations of user and vehicle behavior. We
learn a human-understandable concept layer that we use
to explain sequential driving scenes while learning vehicle control commands. This approach can then be used to
determine whether a change in a preferred gap or steering commands from a human (or autonomous vehicle) is
led by an external stimulus or change in preferences.

### [Interpretable Concept Bottlenecks to Align Reinforcement Learning Agents](https://arxiv.org/pdf/2401.05821)

### [The relational bottleneck as an inductive bias for efficient abstraction](https://www.sciencedirect.com/science/article/pii/S1364661324000809)

### [A Survey on Information Bottleneck](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10438074)

### [Auxiliary Losses for Learning Generalizable Concept-based Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/555479a201da27c97aaeed842d16ca49-Paper-Conference.pdf)

### [Learning to Receive Help:Intervention-Aware Concept Embedding Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/770cabd044c4eacb6dc5924d9a686dce-Paper-Conference.pdf)

### [BREAKING THE ATTENTION BOTTLENECK](https://arxiv.org/pdf/2406.10906)

### [Incremental Residual Concept Bottleneck Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Shang_Incremental_Residual_Concept_Bottleneck_Models_CVPR_2024_paper.pdf)


### [CONCEPT BOTTLENECK MODELS WITHOUT PREDEFINED CONCEPTS](https://arxiv.org/pdf/2407.03921)
There has been considerable recent interest in interpretable concept-based models such as Concept
Bottleneck Models (CBMs), which first predict human-interpretable concepts and then map them
to output classes. To reduce reliance on human-annotated concepts, recent works have converted
pretrained black-box models into interpretable CBMs post-hoc. However, these approaches predefine
a set of concepts, assuming which concepts a black-box model encodes in its representations. In this
work, we eliminate this assumption by leveraging unsupervised concept discovery to automatically
extract concepts without human annotations or a predefined set of concepts. We further introduce an
input-dependent concept selection mechanism that ensures only a small subset of concepts is used
across all classes.

### [Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models](https://arxiv.org/pdf/2405.01531?)
existing approaches often require
numerous human interventions per image to achieve strong performances, posing practical challenges in
scenarios where obtaining human feedback is expensive. In this paper, we find that this is noticeably
driven by an independent treatment of concepts during intervention, wherein a change of one concept
does not influence the use of other ones in the model’s final decision. To address this issue, we introduce
a trainable concept intervention realignment module, which leverages concept relations to realign concept
assignments post-intervention. Across standard, real-world benchmarks, we find that concept realignment
can significantly improve intervention efficacy; significantly reducing the number of interventions needed
to reach a target classification performance or concept prediction accuracy. In addition, it easily integrates
into existing concept-based architectures without requiring changes to the models themselves.

### [Diverse Concept Proposals for Concept Bottleneck Models](https://arxiv.org/pdf/2412.18059)
They identify a small number of humaninterpretable concepts in the data, which they then
use to make predictions. Learning relevant concepts from data proves to be a challenging task.
The most predictive concepts may not align with
expert intuition, thus, failing interpretability with
no recourse. Our proposed approach identifies
a number of predictive concepts that explain the
data. By offering multiple alternative explanations, we allow the human expert to choose the
one that best aligns with their expectation. To
demonstrate our method, we show that it is able
discover all possible concept representations on a
synthetic dataset.

### [FAITHFUL VISION-LANGUAGE INTERPRETATION VIA CONCEPT BOTTLENECK MODELS](https://openreview.net/pdf?id=rp0EdI8X4e)
Labelfree CBMs have emerged to address this, but they remain unstable, affecting their
faithfulness as explanatory tools. To address this issue of inherent instability, we
introduce a formal definition for an alternative concept called the Faithful VisionLanguage Concept (FVLC) model. We present a methodology for constructing
an FVLC that satisfies four critical properties. Our extensive experiments on four
benchmark datasets using Label-free CBM model architectures demonstrate that
our FVLC outperforms other baselines regarding stability against input and concept set perturbations. Our approach incurs minimal accuracy degradation compared to the vanilla CBM, making it a promising solution for reliable and faithful
model interpretation.

### [Beyond Concept Bottleneck Models:How to Make Black Boxes Intervenable?](https://arxiv.org/pdf/2401.13544)
An advantage of this model class is the user’s ability to intervene on predicted concept values, affecting the downstream output. In this work,
we introduce a method to perform such concept-based interventions on pretrained
neural networks, which are not interpretable by design, only given a small validation set with concept labels. Furthermore, we formalise the notion of intervenability
as a measure of the effectiveness of concept-based interventions and leverage this
definition to fine-tune black boxes. Empirically, we explore the intervenability of
black-box classifiers on synthetic tabular and natural image benchmarks. We focus
on backbone architectures of varying complexity, from simple, fully connected
neural nets to Stable Diffusion. We demonstrate that the proposed fine-tuning improves intervention effectiveness and often yields better-calibrated predictions

### [ENERGY-BASED CONCEPT BOTTLENECK MODELS: UNIFYING PREDICTION, CONCEPT INTERVENTION, AND PROBABILISTIC INTERPRETATIONS](https://arxiv.org/pdf/2401.14142)
https://github.com/xmed-lab/ECBM
CUB: pcbm 63.5 this 81.2
(1) they often fail to capture the high-order, nonlinear interaction between concepts, e.g., correcting a predicted concept (e.g., “yellow breast”) does not help correct highly correlated concepts (e.g., “yellow belly”), leading to suboptimal final accuracy; (2) they cannot
naturally quantify the complex conditional dependencies between different concepts and class labels (e.g., for an image with the class label “Kentucky Warbler”
and a concept “black bill”, what is the probability that the model correctly predicts another concept “black crown”), therefore failing to provide deeper insight
into how a black-box model works. In response to these limitations, we propose
Energy-based Concept Bottleneck Models (ECBMs). Our ECBMs use a set
of neural networks to define the joint energy of candidate (input, concept, class)
tuples. 

### [CLIP-QDA: An Explainable Concept Bottleneck Model](https://arxiv.org/pdf/2312.00110)
In this paper, we introduce an explainable algorithm designed from a multi-modal foundation
model, that performs fast and explainable image classification. Drawing inspiration from
CLIP-based Concept Bottleneck Models (CBMs), our method creates a latent space where
each neuron is linked to a specific word. Observing that this latent space can be modeled
with simple distributions, we use a Mixture of Gaussians (MoG) formalism to enhance the
interpretability of this latent space. Then, we introduce CLIP-QDA, a classifier that only
uses statistical values to infer labels from the concepts. In addition, this formalism allows for
both sample-wise and dataset-wise explanations. These explanations come from the inner
design of our architecture, our work is part of a new family of greybox models, combining
performances of opaque foundation models and the interpretability of transparent models.

### [CONCEPT BOTTLENECK LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2412.07992)
We introduce the Concept Bottleneck Large Language Model (CB-LLM), a pioneering approach to creating inherently interpretable Large Language Models (LLMs).
Unlike traditional black-box LLMs that rely on post-hoc interpretation methods
with limited neuron function insights, CB-LLM sets a new standard with its built-in
interpretability, scalability, and ability to provide clear, accurate explanations. We
investigate two essential tasks in the NLP domain: text classification and text
generation. In text classification, CB-LLM narrows the performance gap with traditional black-box models and provides clear interpretability. In text generation, we
show how interpretable neurons in CB-LLM can be used for concept detection and
steering text generation. Our CB-LLMs enable greater interaction between humans
and LLMs across a variety of tasks — a feature notably absent in existing LLMs.

### [Learning to Intervene on Concept Bottlenecks](https://arxiv.org/pdf/2308.13453)
 they allow users to perform interventional interactions on these concepts
by updating the concept values and thus correcting the predictive output of the model. Up to this
point, these interventions were typically applied
to the model just once and then discarded. To
rectify this, we present concept bottleneck memory models (CB2Ms), which keep a memory of past interventions. Specifically, CB2Ms leverage
a two-fold memory to generalize interventions to
appropriate novel situations, enabling the model
to identify errors and reapply previous interventions. This way, a CB2M learns to automatically improve model performance from a few initially
obtained interventions.

### [Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts](https://arxiv.org/pdf/2412.14097?)
we focus on the test-time deployment of such an interpretable CBM pipeline “in the wild”,
where the input distribution often shifts from the original training distribution. We first
identify the potential failure modes of such a pipeline under different types of distribution
shifts. Then we propose an adaptive concept bottleneck framework to address these failure
modes, that dynamically adapts the concept-vector bank and the prediction layer based
solely on unlabeled data from the target domain, without access to the source (training)
dataset.

### [Relational Concept Bottleneck Models]()
relational deep learning models, such as Graph Neural Networks (GNNs), are
not as interpretable as CBMs. To overcome these limitations, we propose Relational Concept Bottleneck Models (R-CBMs), a family of relational deep learning
methods providing interpretable task predictions. As special cases, we show that
R-CBMs are capable of both representing standard CBMs and message-passing
GNNs. To evaluate the effectiveness and versatility of these models, we designed a
class of experimental problems, ranging from image classification to link prediction
in knowledge graphs. In particular we show that R-CBMs (i) match generalization performance of existing relational black-boxes, (ii) support the generation
of quantified concept-based explanations, (iii) effectively respond to test-time interventions, and (iv) withstand demanding settings including out-of-distribution
scenarios, limited training data regimes, and scarce concept supervisions.

### [Semi-supervised Concept Bottleneck Models](https://arxiv.org/pdf/2406.18992)
These concept labels are typically provided by
experts, which can be costly and require significant resources and effort. Additionally, concept saliency maps frequently misalign with input saliency maps, causing
concept predictions to correspond to irrelevant input features - an issue related to
annotation alignment. To address these limitations, we propose a new framework
called SSCBM (Semi-supervised Concept Bottleneck Model). Our SSCBM is
suitable for practical situations where annotated data is scarce. By leveraging joint
training on both labeled and unlabeled data and aligning the unlabeled data at the
concept level, we effectively solve these issues. We proposed a strategy to generate
pseudo labels and an alignment loss.

### [A Theoretical design of Concept Sets: improving the predictability of concept bottleneck models](https://openreview.net/pdf?id=oTv6Qa12G0)
In this work, we define concepts within the machine learning context, highlighting their core properties: expressiveness and model-aware inductive bias, and
we make explicit the underlying assumption of CBMs

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
