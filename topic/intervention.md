### [Learning to Receive Help: Intervention‑Aware Concept Embedding Models](https://arxiv.org/pdf/2309.16928)
- Concept-based models “can be helped”, but they don’t know how to receive help.
- Intervention-aware training
- Concept Intervention Policy
- Intervention-aware loss
- TravelingBirds（intervention-heavy）

### [Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models](https://arxiv.org/abs/2405.01531)
- Why do concept interventions in Concept Bottleneck Models often require many corrections to be effective, and how can we make interventions more efficient and reliable?
- Is this inefficiency caused by the way CBMs treat concepts as independent during intervention?
- CBMs assume that intervening on one concept should not affect other concepts.
- Given an intervention, how should other concepts change to stay consistent?

### [CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_CausalVAE_Disentangled_Representation_Learning_via_Neural_Structural_Causal_Models_CVPR_2021_paper.pdf)
- 如何在“非独立语义因素”的前提下，学习可控、可解释、支持干预的潜表示？
- 用结构因果模型（SCM）作为 VAE 的先验，把“解耦”从“统计独立”升级为“因果可分”。

### [Erasing Concepts, Steering Generations: A Comprehensive Survey of Concept Suppression](https://arxiv.org/pdf/2505.19398)
- 在文本到图像（T2I）生成模型中，如何“选择性地抑制（erase / suppress）特定语义概念”，同时尽可能保持模型的整体生成能力与其他概念不受影响？
- Intervention Level: (1) Text Encoder Level,修改 CLIP embedding,过滤或旋转特定语义方向;(2)Cross-Attention Level,控制 token ↔ image 的对齐;(3)U-Net / Diffusion Backbone,直接改模型权重
- Optimization Structure: (1) Loss-based optimization,加 suppression loss;(2)Closed-form projection,线性空间移除;(3)Adapter / plug-in,LoRA / side modules;(4)Adversarial training,提高鲁棒性
- Semantic Scope: (1) 单一概念（object / style）; (2)多概念; (3) 概念组合（hard case）

### [Concept Lancet: Image Editing with Compositional Representation Transplant](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Concept_Lancet_Image_Editing_with_Compositional_Representation_Transplant_CVPR_2025_paper.pdf)
- 在扩散模型的图像编辑中，如何在不依赖人工调参的情况下，自动决定“合适的编辑强度”，从而实现既有效又不破坏原图一致性的概念级编辑？编辑失败并非方向不对，而是不知道“该改多少”。
- Concept Lancet (CoLan) 一个 zero-shot、plug-and-play 的 概念表示移植（representation transplant） 框架。在 latent 空间中 显式估计概念含量;根据任务（add / remove / replace）;精确调整概念分量，而非盲目加 direction

### [ACE: Concept Editing in Diffusion Models without Performance Degradation](https://arxiv.org/pdf/2503.08116)
- 在扩散模型中进行概念编辑（尤其是擦除不安全概念）时，如何在“成功移除目标概念”的同时，避免模型整体生成能力和语义一致性的退化？
- Cross Null‑Space Projection只在“不会影响模型正常能力的子空间”里修改模型。编辑对象：Cross‑Attention 层

### [Unified Concept Editing in Diffusion Models](https://arxiv.org/html/2308.14761)
- 在文本到图像扩散模型中，如何用一个统一的方法，同时对多种不同类型的概念（偏见、风格、NSFW 等）进行编辑，并且能够规模化地并发修改多个概念，而不需要重新训练模型？
- 把所有“概念编辑”问题，统一转化为对 cross‑attention 线性投影的闭式修改。不训练,闭式解（closed‑form）,一次性并发编辑多个概念
- 编辑的对象：Cross‑Attention 投影矩阵
- 条件化线性回归（Closed‑form）

### []()

### []()

### []()

### []()
