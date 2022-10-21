## Background

The prompt is to add a form or pattern to help the model remember what they learned in the pre-training phase. 
For instance, to classify the sensitive label of the sentence [X] "I love this movie.", we could add a prefix prompt at the end, i.e. "I love this movie. Overall it was a [Z] movie." Then ask the network to fill in the word [Z]. The words, like excellent, great, and wonderful, will mapping to positive labels.

**How to design prompt?**
1. The shape of Prompt
-  number and position of [X] and [Z]
2. Design Prompt manually
- LAMA dataset, Petroni et al. gave the cloze temples
- Brown et al designes prefix templates
3.  Learnt Prompt automanually
- Discrete Prompts
  - Prompt Mining: Find the path for a given input [X] and output [Y], and use the most frequent middle words as a prompt.
  - Prompt Paraphrasing: Transform existing seeds of prompts into prompts candidating. For instance, translate prompts to another language and then translate back; find the synonyms.
  - Gradient-based Search: Use gradient descent to find the words and the best combination for the prompt.
  - Prompt Generation: Use text generator, for instance, Gao et al. use T5; Ben-David et al. use domain adaptive algorithm.
  - Prompt Scoring: Take the techniques from the knowledge graph, and use (entity, relation, entity) as a pattern, using a bi-direction language model to rate these prompts.
- Continuous Prompts
  - Prefix Tuning:
  - Tuning Initialized with Discrete Prompts
  - Hard-Soft Prompt Hybrid Tuning

### [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586)

[S.O.T.A.](http://pretrain.nlpedia.ai/)

### [On the Robustness of Dialogue History Representation in Conversational Question Answering: A Comprehensive Study and a New Prompt-based Method](https://arxiv.org/abs/2206.14796)

### [Black-box Prompt Learning for Pre-trained Language Models](https://arxiv.org/pdf/2201.08531.pdf)

The algorithm to find Prompt shares similarity to adversarial attacks.

### [Are Sample-Efficient NLP Models More Robust?](https://arxiv.org/pdf/2210.06456.pdf)

## Visual

### [Visual Prompting for Adversarial Robustness](https://arxiv.org/pdf/2210.06284.pdf)
