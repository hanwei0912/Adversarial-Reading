## Robustness

### [Amplitude-Phase Recombination: Rethinking Robustness of Convolutional Neural Networks in Frequency Domain](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Amplitude-Phase_Recombination_Rethinking_Robustness_of_Convolutional_Neural_Networks_in_Frequency_ICCV_2021_paper.html)
In this paper, we notice that the CNN tends to converge at the local optimum which is closely related to the high-frequency components of the training images, while the amplitude spectrum is easily disturbed such as noises or common corruptions. In contrast, more empirical studies found that humans rely on more phase components to achieve robust recognition. This observation leads to more explanations of the CNN's generalization behaviors in both robustness to common perturbations and out-of-distribution detection, and motivates a new perspective on data augmentation designed by re-combing the phase spectrum of the current image and the amplitude spectrum of the distracter image. That is, the generated samples force the CNN to pay more attention to the structured information from phase components and keep robust to the variation of the amplitude.

A way to augment data
