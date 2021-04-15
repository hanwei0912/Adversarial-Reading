### [MITIGATING ADVERSARIAL EFFECTS THROUGH RANDOMIZATION](https://arxiv.org/pdf/1711.01991.pdf)

- utilize randomization at inference time to mitigate adversarial effects
	1) random resizing
	2) random padding
- very effective at defending against both single-step and iterative attackse

### [STOCHASTIC ACTIVATION PRUNING FOR ROBUST ADVERSARIAL DEFENSE](https://arxiv.org/pdf/1803.01442.pdf)
- take inspiration from game theory and cast the problem as a minimax zero-sum game
- the optimal strategy for both players requires a stochasic policy
- propose Stochatic Activation Pruning
	1) prunes random subset of activations
	2) scales up the survivors to conpensate


### [CERTIFIED DEFENSES FOR ADVERSARIAL PATCHES](https://arxiv.org/pdf/2003.06693.pdf)
- against patch attacks
- start by examining existing defense strategies
	1) easily be broken in white-box attacks


### [Barrage of Random Transforms for Adversarially Robust Defense](https://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf)

- stochastically combining a large number of individually week defenses into a single barrage of randomized transformation to build a strong defense against adversarial attack
- Barrage of Random Transforms
	- take pre-trained model and randomly pick k transformations to apply to each images and perform an additional 100 epochs of training so that the network familiar with the transformation
	- one have a trained model: advarsary attack it
		1) reduce the top-1 accuracy
		2) reduce the top-5 accuracy
		3) increase the targeted success rate


