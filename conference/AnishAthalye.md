## [Anish Athalye](https://www.anish.io/)

### Attack

#### [Synthesizing Robust Adversarial Examples](https://arxiv.org/abs/1707.07397)

- robust 3D adversarial objects (turtle paper)
- present the first algorithm for synthesizing examples that are adversarial over a chosen distribution of transformations. We synthesize two-dimensional adversarial images that are robust to noise, distortion, and affine transformation. We apply our algorithm to complex three-dimensional objects, using 3D-printing to manufacture the first physical adversarial objects.

#### [Black-box Adversarial Attacks with Limited Queries and Information](https://arxiv.org/abs/1804.08598)

- three realistic threat models that more accurately characterize many real-world classifiers: the query-limited setting, the partial-information setting, and the label-only setting
- new attacks that fool classifiers under these more restrictive threat models, where previous methods would be impractical or ineffective
- Natural evolutionary strategies (query-limited setting): use NES to estimate the gradient. choose a search distribution of random gaussian noise around the current images, employ antithetic sampling to generate a population.
- mathmatically formulate the partial-information setting and label-only setting, and propose partial information attack. 

### Defense

#### [Notary: A Device for Secure Transaction Approval](https://pdos.csail.mit.edu/papers/notary:sosp19.pdf)

- hardware and software architecture for
running isolated approval agents in the form factor of a USB
stick with a small display and buttons. Approval agents allow factoring out critical security decisions, such as getting
the user’s approval to sign a Bitcoin transaction or to delete
a backup, to a secure environment. The key challenge addressed by Notary is to securely switch between agents on
the same device. Prior systems either avoid the problem by
building single-function devices like a USB U2F key, or they
provide weak isolation that is susceptible to kernel bugs,
side channels, or Rowhammer-like attacks. Notary achieves
strong isolation using reset-based switching, along with the
use of physically separate systems-on-a-chip for agent code
and for the kernel, and a machine-checked proof of both
the hardware’s register-transfer-level design and software,
showing that reset-based switching leaks no state. Notary
also provides a trustworthy I/O path between the agent code
and the user, which prevents an adversary from tampering
with the user’s screen or buttons.

#### [pASSWORD tYPOS and How to Correct Them Securely](https://www.ieee-security.org/TC/SP2016/papers/0824a799.pdf)

- We provide the first treatment of typo-tolerant
password authentication for arbitrary user-selected passwords.
Such a system, rather than simply rejecting a login attempt with
an incorrect password, tries to correct common typographical
errors on behalf of the user. Limited forms of typo-tolerance
have been used in some industry settings, but to date there has
been no analysis of the utility and security of such schemes.

### Analysis

#### [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://arxiv.org/abs/1802.00420)
- BPDA

#### [Evaluating and Understanding the Robustness of Adversarial Logit Pairing](https://arxiv.org/abs/1807.10272)

- constribution: (1)*Robustness*: Under the white-box targeted attack threat model specified in Kannan et al.,
we upper bound the correct classification rate of the defense to 0.6% (Table 1). We also
perform targeted and untargeted attacks and show that the attacker can reach success rates
of 98.6% and 99.9% respectively (Figures 1, 2). (2) *Formulation*:We analyze the ALP loss function and contrast it to that of Madry et al.,
pointing out several differences from the robust optimization objective (Section 4.1). (3) *loss landscape*: We analyze the loss landscape induced by ALP by visualizing loss landscapes and adversarial attack trajectories (Section 4.2).

- trained on natural images vs. adversarial images; generating targeted adversarial examples
- loss landscape:  We vary the input
along a linear space defined by the sign of the gradient and a random Rademacher vector, where
the x and y axes represent the magnitude of the perturbation added in each direction and the z axis
represents the loss. The plots provide evidence that ALP sometimes induces a “bumpier,” depressed
loss landscape tightly around the input points.
- better than baseline, but not robust under the considered threat model.

#### [On the Robustness of the CVPR 2018 White-Box Adversarial Example Defenses](https://arxiv.org/abs/1804.03286)

- Evaluation on "Pixel Delection" and "High-level representation guided denoiser"

#### [On Evaluating Adversarial Robustness](https://arxiv.org/abs/1902.06705)
- In this paper, we discuss the methodological foundations, review commonly accepted best practices, and suggest new methods
for evaluating defenses to adversarial examples.
