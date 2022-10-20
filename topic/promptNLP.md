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
  - Prompt Paraphrasing: 
  - Gradient-based Search
  - Prompt Generation
  - Prompt Scoring  
- Continuous Prompts
  - Prefix Tuning
  - Tuning Initialized with Discrete Prompts
  - Hard-Soft Prompt Hybrid Tuning
