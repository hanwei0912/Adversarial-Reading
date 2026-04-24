### [Towards Robust Interpretability with Self‑Explaining Neural Networks (SENN)](https://arxiv.org/abs/1806.07538)

Self‑Explaining Neural Networks (Alvarez‑Melis & Jaakkola, NeurIPS 2018) propose intrinsically interpretable models that enforce robustness as stability of explanations under small perturbations, laying early conceptual foundations for later work on robustness in concept‑based models.

### [Understanding and Enhancing Robustness of Concept‑Based Models](https://arxiv.org/pdf/2211.16080)

Sinha et al. (AAAI 2023) provide the first systematic analysis of adversarial robustness in concept‑based models, demonstrating that interpretability alone does not guarantee robustness and that concepts may introduce new vulnerabilities. By designing concept‑aware adversarial attacks and applying adversarial training at the concept level, they show that robustness can be partially improved, while revealing a fundamental tension between concept interpretability and adversarial stability.

Metrics:(A) Concept Accuracy under Attack; (B) Concept Stability / Consistency: overlap‑based metrics, notably Jaccard similarity; (C) Prediction–Concept Decoupling

Concept Fragility (Concepts flip easily; Explanations change without prediction change): weak concept supervision; low‑margin concept classifiers; disentangled but brittle representations

### [Language Guided Concept Bottleneck Models for Interpretable Continual Learning](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Language_Guided_Concept_Bottleneck_Models_for_Interpretable_Continual_Learning_CVPR_2025_paper.pdf)

1. Vanilla CBMs and SENNs are more vulnerable to adversarial attacks than comparable end‑to‑end models;
2. Concept bottlenecks create new attack surfaces;
3. Attackers can: flip concepts while preserving labels, or preserve labels while corrupting explanations.

### [Exploring the Impact of Conceptual Bottlenecks on Adversarial Robustness](https://ieeexplore.ieee.org/document/10677449)

This paper explicitly tests a hypothesis that was implicitly questioned in earlier works: Because CBMs compress inputs into high‑level concepts, they may filter out non‑robust features and thus be more adversarially robust.
CBMs—especially sequentially trained CBMs—often achieve higher adversarial accuracy than standard CNNs under the same attack settings. This holds: For weak–moderate attacks; Without adversarial training; More consistently when concept dimensionality is small.
Conceptual bottlenecks act as a regularizer / information filter, removing non‑essential variations and thereby improving robustness.
While Sinha et al. (2023) demonstrate that concept bottlenecks introduce new vulnerabilities at the level of concept predictions, Rasheed et al. (2024) provide complementary evidence that such bottlenecks may still improve adversarial robustness at the output level under certain training regimes.


