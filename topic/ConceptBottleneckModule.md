## 2024

### [Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification](https://arxiv.org/pdf/2411.05698?)

### [On the Concept Trustworthiness in Concept Bottleneck Models](https://arxiv.org/abs/2403.14349)
- concept trustworthness score  **We need to check it**
- three modules: cross-layer alignment, prediction alignment, cross-image alignment
https://github.com/hqhQAQ/ProtoCBM
CUB: pcbm 58.8 this 68.3

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

## 2023

### [A Closer Look at the Intervention Procedure of Concept Bottleneck Models](https://proceedings.mlr.press/v202/shin23a/shin23a.pdf)

### [Probabilistic Concept Bottleneck Models](https://arxiv.org/pdf/2306.01574)


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
