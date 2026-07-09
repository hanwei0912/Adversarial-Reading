# Mechanistic Interpretability

''Try to reverse-engineer a transformer into human-understandable algorithms.''

- What representations exist?
- What circuits (sub-algorithms) compute those outputs?
- How is information stored, transformed, and used?

## Transformer = linear-algebraic circuit
Goal: treat transformer as composable linear circuits

### [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- 2021
- This is the foundation of the entire field.
- Main idea: Treat the transformer as a sum of linear operations over a shared channel (residual stream).
*Residual stream*:
- One vector space where everything communicates
- All layers read from and write to it
*Attention heads = independent modules*:
- Each head reads via QK, writes via OV
- Heads are additive and separable
*Path / circuit viewpoint*:
Model output can be decomposed into:
- paths through heads and layers
- each path = partial computation
*Induction heads*
- Special heads that implement copying pattern
- foundation of in-context learning

### [LINEARITY OF RELATION DECODING IN TRANSFORMER LANGUAGE MODELS](https://arxiv.org/pdf/2308.09124)
- 2023
- Main idea: Semantic relations (e.g. Paris $\to$ France) are linearly decodable from hidden states
- Meaning: relationships are not complex nonlinear objects; they live in linear subspaces
- Many high-level concepts are surprisingly linearly structured.

### [Jump to Conclusions: Short-Cutting Transformers with Linear Transformations](https://arxiv.org/pdf/2303.09435)
- 2023
- Main idea: You can approximate LLM behavior by: applying simple linear transformations; skipping layers
- Interpretation: much of computation is already decided early, later layers just refine

## “Lens” methods: peek into hidden states
Goal: decode what the model “believes” at each layer

### [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- 2020
- Main idea: Apply the final output decoder (unembedding) to intermediate layers. This gives "what token the model is predicting at layer L"
- Predictions are formed early and refined across layers.
- Instead of “model builds representation → then predicts”, it shows "model predicts immediately, then refines"

### [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/pdf/2303.08112)
- 2023
- Motivation: Fix logit lens problems of representation drift and poor early-layer decoding
- Method: Train a small linear probe per layer: $h_l \to$ affine transform $\to$ final space $\to$ logits
- Key result: Represents layer-wise prediction trajectory; smoother convergence to final prediction
*New concept: Prediction trajectory*
- each layer refines the distribution
- behaves like iterative inference
- applications: detect malicious inputs; understand reasoning evolution

### [Future Lens: Anticipating Subsequent Tokens from a Single Hidden State](https://aclanthology.org/2023.conll-1.37/)
- 2023
- Question: Does a hidden state contain info about future tokens beyond next token?
- Result: Yes, single hidden state can predict multiple future tokens (~48% accuracy)
- Implication: representations are information-rich; knowledge is distributed and anticipatory
- hidden state $\neq$ “next-token only”, it encodes entire continuation



## Internal cognition / high-level structure
Goal: identify global structures of reasoning 

### [Progress Update #1 from the GDM Mech Interp Team](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/summary-progress-update-1-from-the-gdm-mech-interp-team)
- This is not a technical paper — it gives: research agenda, open problems, experimental directions
- Key themes: circuit discovery, scaling interpretability, automated methods
- Field is transitioning from manual analysis → scalable science.

### [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
- 2026
- Main idea: LLMs develop a special subspace (J-space):small set of representations; accessible, controllable, reportable
- Properties: These representations, correspond to concepts, can be verbalized, are used in reasoning, can be intentionally controlled


## Others

### [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- Identifies induction heads: attention heads implementing pattern copying
- Mechanism: detect pattern [A] ... [A] → predict next token after previous A
- Provides strong evidence they explain in-context learning (ICL)

### [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- Explains polysemantic neurons
- Introduces superposition: many features stored in same dimension; causes neurons to be “messy”
- Shows phase transitions and geometry effects

### [Sparse Autoencoders Can Interpret Randomly Initialized Transformers](https://arxiv.org/abs/2501.17727)
### [A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of LLMs](https://aclanthology.org/2025.findings-emnlp.89/)
### [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
### [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)
### [Understanding Scaling Laws with Statistical and Approximation Theory for Transformer Neural Networks](https://arxiv.org/abs/2411.06646)














### []()
