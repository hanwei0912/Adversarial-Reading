### [PointNet](https://arxiv.org/abs/1612.00593)

### [MV3D](https://arxiv.org/abs/1611.07759)
- 2017
- input: LIDAR bird view + LIDAR front view + Image（RGB）
- conv-based fusion architecture

### [AVOD](https://arxiv.org/abs/1712.02294)
- 2017
- improving MV3D ( remove the front view, use FPN for feature extraction; use crop and resize when merging different branches; add constrain on 3D bounding box

### [PointNet++]()

### [Frustum PointNet]()



### [VoxelNet](https://arxiv.org/abs/1711.06396)
- 2018 CVPR
- end-to-end: feature learning network -> convolutional middle layers (3D CNN) -> region proposal network
- learning feature network: voxel partition; grouping; random sampling; stacked voxel feature encoding; sparse 4D tensor



### [SECOND](https://www.mdpi.com/1424-8220/18/10/3337)
- 2018
- improve VoxelNet with 3D sparsely embedded convolutional operation
- point cloud -> voxel features and coordinates -> voxel feature extractor -> sparse conv layers -> RPN


### [Pointpillar](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf)
- 2019 CVPR
- point clouds -> pillar feature net -> backbone (2D cnn) -> detection head (SSD) -> prediction
- pillar feature net: stacked pillars (grid/ pillar index) -> learned features -> pseudo image
  
### [Votenet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qi_Deep_Hough_Voting_for_3D_Object_Detection_in_Point_Clouds_ICCV_2019_paper.pdf)
- 2019 ICCV
- point cloud feature learning backbone + deep hough voting for box proposal
- use network to represent and select interest points and learn the vote aggregation


### [PV-RCNN++](https://arxiv.org/pdf/2102.00463.pdf)
- 2020 CVPR
- combine point-based method and voxel-based method(the authors claim it is the same as grid-based methods)
- raw-points --> 3D voxel --> Keypoint --> Grid Point --> Refine
- two stage: voxel-to-keypoint scene encoding step (proposals); keypoint-to-grid RoI feature abstraction (refine)
- stage 1: extract multi-scale features of voxel convolution; voxel set abstraction module
- stage 2: RoI-grid pooling module

### [CenterPoint]()
- 2021 CVPR
- point cloud --(3D backbone)--> map-view features --(head)--> First stage: Centers and 3D boxes --(MLP)--> Second stage: scores and 3D boxes
- Backbone 3D(mean VFE + VoxelRes_Bx) -> Backbone 2D (Height Compression + BaseBEV backbone) --> DenseHead (CenterHead)
