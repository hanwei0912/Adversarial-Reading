### [CAUSAL DISCOVERY WITH REINFORCEMENT LEARNING](https://arxiv.org/pdf/1906.04477.pdf)

- input: random sampling of data x, and reshape them to s
- output: binary adjacency matrix
- encoder: transfermer, encoder a sample to a latent code
- decoder: g_ij(W1,W2,u) = uˆT tanh(W1 enc_i + W2 enc_j), \sigma (g_ij), which gives the edge from x_i to x_j
- then using RL to search the adjacency matrix
- score function ==> bayeisan information Criteria -> maximum likelihood == mimumum least sqare loss
- award function ==> score function + acyclic punish term
- score adjustment by min-max normalization
