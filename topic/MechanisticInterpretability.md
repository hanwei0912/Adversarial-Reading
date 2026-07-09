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


### [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)



### [Progress Update #1 from the GDM Mech Interp Team](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/summary-progress-update-1-from-the-gdm-mech-interp-team)





### [Jump to Conclusions: Short-Cutting Transformers with Linear Transformations](https://arxiv.org/pdf/2303.09435)





### []()
