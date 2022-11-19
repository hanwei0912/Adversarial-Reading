## Backdoor Attack at Data Collection Stage

Inject only a smaller number of poison data into the training set to create a backdoor model:

1. [Targeted backdoor attacks on deep learning systems using data poisoning](https://arxiv.org/pdf/1712.05526.pdf)
- 2017
(1) the adversary has no knowledge of the model and the training set used by the victim system; (2) the attacker is allowed to inject only a small amount of poisoning samples; (3) the backdoor key is hard to notice even by human beings to achieve stealthiness. 

Generate poison data consisting of the perturbed images and the corresponding correct labels

1. [Poison frogs! targeted clean-label poisoning attacks on neural networks]()
- 2018 neural networks

2. [Hidden trigger backdoor attacks]()
- 2020 AAAI

Backdoor triggers in the frequency domain:
1. [Rethinking the backdoor attacks’ triggers: A frequency perspective]()
- 2021 ICCV

## Backdoor Attack at Model Training Stage

Outsourced training can cause security risk via altering training data: the imperceptibility of the trigger patterns are critical to the success of
backdoor attack.
1. [Badnets: Evaluating backdooring attacks on deep neural networks]()
- 2019
2. [Input-aware dynamic backdoor attack]()
- 2020 Advances in Neural Information Processing Systems
3. [Wanet-imperceptible warping-based backdoor attack]()
- 2021 ICLR 
4. [Lira: Learnable, imperceptible and robust backdoor attacks]()
- 2021 ICCV
designs a novel backdoor attack framework, LIRA, which
learns the optimal imperceptible trigger injection function to poison the input

5. [Invisible backdoor attacks on deep neural networks via steganography and regularization]()
- 2020 IEEE TDSC
is proposed to
generate backdoor images via subtle image warping

6. [Backdoor attack with imperceptible input and latent modification]()
- 2021 ANIPS
achieves high attack success rate via generating imperceptible input noise which is stealthy in both the input and latent
spaces

## Backdoor Attack at Model Compression Stage

1. [Stealthy backdoors as compression artifacts]()
- 2021
- proposes a method to inject inactive backdoor to the fullsize model, and the backdoor will be activated after the model is compressed.

2. [Quantization backdoors to deep learning models]()
- 2021
- discovers that the standard quantization operation can be abused to enable
backdoor attacks. 

3. [Qu-anti-zation: Exploiting quantization artifacts for achieving adversarial outcomes]()
- 2021 ANIPS
- propose to use quantization-aware backdoor training to ensure the effectiveness of backdoor if model is further quantized

## Backdoor Defense

Detection-style methods: aim to identify the potential malicious training samples via statistically analyzing
some important behaviors of models 
1. [ Detecting backdoor attacks on deep neural networks by activation clustering]()
- 2018

2. [ Spectral signatures in backdoor attacks]()
- 2018 ANIPS

5. [ Strip: A defence against trojan attacks on deep neural networks]()
- 2019 ACSAC
- use activation values

6. [Fine-pruning: Defending against backdooring attacks on deep neural networks]()
-  2018 SRAID
- use predictions 

Performing pre-processing of the input, data mitigation-style strategy: targets to eliminate or mitigate the affect of
the backdoor triggers, so the infected model can still behave normally with the
presence of trigger-contained inputs
1. [Neural trojans]()
- 2017 IEEE ICCD
2. [Rethinking the trigger of backdoor attack]()
- 2020

Model modification approaches:

Re-training on clean data:
1. [Bridging mode connectivity in loss landscapes and adversarial robustness]()
- 2020

Pruning the infected neurons
1. [Fine-pruning: Defending against backdooring attacks on deep neural networks.]()
-  2018 SRAID

2. [Adversarial neuron pruning purifies backdoored deep models]()
- 2021 ANIPS
