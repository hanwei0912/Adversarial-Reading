## Defenses (adv. training)

### [__Scalable Verified Training for Provably Robust Image Classification__](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gowal_Scalable_Verified_Training_for_Provably_Robust_Image_Classification_ICCV_2019_paper.pdf)


### [__Defending Against Universal Perturbations With Shared Adversarial Training__](http://openaccess.thecvf.com/content_ICCV_2019/papers/Mummadi_Defending_Against_Universal_Perturbations_With_Shared_Adversarial_Training_ICCV_2019_paper.pdf)

1. abstract
In this work, we show that adversarial training is
more effective in preventing universal perturbations, where
the same perturbation needs to fool a classifier on many
inputs. Moreover, we investigate the trade-off between robustness against universal perturbations and performance
on unperturbed data and propose an extension of adversarial training that handles this trade-off more gracefully.


* http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Bilateral_Adversarial_Training_Towards_Fast_Training_of_More_Robust_Models_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.pdf

Defenses (loss functions)

* http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Improving_Adversarial_Robustness_via_Guided_Complement_Entropy_ICCV_2019_paper.pdf

* http://openaccess.thecvf.com/content_ICCV_2019/papers/Mustafa_Adversarial_Defense_by_Restricting_the_Hidden_Space_of_Deep_Neural_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhong_Adversarial_Learning_With_Margin-Based_Triplet_Embedding_Regularization_ICCV_2019_paper.pdf


## Defenses (denoising)

### [CIIDefence: Defeating Adversarial Attacks by Fusing Class-specific Image Inpainting and Image Denoising](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gupta_CIIDefence_Defeating_Adversarial_Attacks_by_Fusing_Class-Specific_Image_Inpainting_and_ICCV_2019_paper.pdf)

The proposed defence mechanism is inspired by the recent works mitigating the adversarial disturbances by the means of image reconstruction and denoising. However, unlike the previous works, we apply the reconstruction only for small and carefully selected image areas that are most influential to the current classification outcome. The selection process is guided by the class activation map responses obtained for multiple top-ranking class labels. The same regions are also the most prominent for the adversarial perturbations and hence most important to purify. The resulting inpainting task is substantially more tractable than the full image reconstruction, while still being able to prevent the adversarial attacks.

__Removing class-specifiac areas__: 找到local maxima点，然后把以该点为中心w窗口内的点都清空。

__BPDA__： 在backward pass的时候把不可导的inpainting，denoising，fusion改成identity。

_________________________

### [Hilbert-based Generative Defense for Adversarial Examples](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bai_Hilbert-Based_Generative_Defense_for_Adversarial_Examples_ICCV_2019_paper.pdf)

To defend against such adversarial perturbations, recently developed PixelDefend purifies a perturbed image based on PixelCNN in a raster scan order (row/column by row/column). However, such scan mode insufficiently exploits the correlations between pixels, which further limits its robustness performance. Therefore, we propose a more advanced Hilbert curve scan order to model the pixel dependencies in this paper. Hilbert curve could well preserve local consistency when mapping from 2-D image to 1-D vector, thus the local features in neighboring pixels can be more effectively modeled. Moreover, the defensive power can be further improved via ensembles of Hilbert curve with different orientations.

[__PixelCNN__](https://zhuanlan.zhihu.com/p/25299749):
现在需要构建一个模型来实现生成一张图片，那么最简单的想法就是一个像素点一个像素点的进行生成，同时将前面生成的像素点作为参考。相当于将预测一张图上所有像素点的联合分布转换为对条件分布的预测。

[__PixelDefend__](https://arxiv.org/pdf/1710.10766.pdf):
使用PixelCNN生成模型将潜在的对抗样本投射回数据流形上，然后将其馈送到分类器中。 作者认为，对抗样本主要在于数据分布的低概率区域。 PixelDefend在分类之前通过使用贪婪的解码程序来“净化”对抗扰动的图像，以近似地找到输入图像的给定范围中的最高概率示例。

____________________

## Defenses (model)

### [Adversarial Robustness vs. Model Compression, or Both?](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Adversarial_Robustness_vs._Model_Compression_or_Both_ICCV_2019_paper.pdf)




## Defenses (tasks other than classification)

http://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Evaluating_Robustness_of_Deep_Image_Super-Resolution_Against_Adversarial_Attacks_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Towards_Adversarially_Robust_Object_Detection_ICCV_2019_paper.pdf

__________________________________________________________
_______________________________________________________

### Attacks (white-box)

* http://openaccess.thecvf.com/content_ICCV_2019/papers/Finlay_The_LogBarrier_Adversarial_Attack_Making_Effective_Use_of_Decision_Boundary_ICCV_2019_paper.pdf

* http://openaccess.thecvf.com/content_ICCV_2019/papers/Ganeshan_FDA_Feature_Disruptive_Attack_ICCV_2019_paper.pdf

* http://openaccess.thecvf.com/content_ICCV_2019/papers/Croce_Sparse_and_Imperceivable_Adversarial_Attacks_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Enhancing_Adversarial_Example_Transferability_With_an_Intermediate_Level_Attack_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Han_Once_a_MAN_Towards_Multi-Target_Attack_via_Learning_Multi-Target_Adversarial_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Joshi_Semantic_Adversarial_Attacks_Parametric_Transformations_That_Fool_Deep_Classifiers_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Afifi_What_Else_Can_Fool_Deep_Learning_Addressing_Color_Constancy_Errors_ICCV_2019_paper.pdf


### Attacks (black-box)

* http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_A_Geometry-Inspired_Decision-Based_Attack_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Brunner_Guessing_Smart_Biased_Sampling_for_Efficient_Black-Box_Adversarial_Attacks_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_On_the_Design_of_Black-Box_Adversarial_Examples_by_Leveraging_Gradient-Free_ICCV_2019_paper.pdf


### Attacks (universal)

http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Universal_Adversarial_Perturbation_via_Prior_Driven_Uncertainty_Approximation_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Universal_Perturbation_Attack_Against_Image_Retrieval_ICCV_2019_paper.pdf


### Attacks (model)

http://openaccess.thecvf.com/content_ICCV_2019/papers/Rakin_Bit-Flip_Attack_Crushing_Neural_Network_With_Progressive_Bit_Search_ICCV_2019_paper.pdf


### Attacks (tasks other than classification)

* http://openaccess.thecvf.com/content_ICCV_2019/papers/Tolias_Targeted_Mismatch_Adversarial_Attack_Query_With_a_Flower_to_Retrieve_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Wiyatno_Physical_Adversarial_Textures_That_Fool_Visual_Object_Tracking_ICCV_2019_paper.pdf

http://openaccess.thecvf.com/content_ICCV_2019/papers/Ranjan_Attacking_Optical_Flow_ICCV_2019_paper.pdf
