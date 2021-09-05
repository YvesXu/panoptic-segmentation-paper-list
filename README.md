## Panoptic Segmentation Paper List

A paper list of panoptic segmentation using deep learning. Sorted by reverse order arXiv number.

Inspired by [deep learning_object_detection](https://github.com/hoya012/deep_learning_object_detection) and [awesome-panoptic-segmentation](https://github.com/Angzz/awesome-panoptic-segmentation).

*Last updated: 2021/09/05*

Update log
- <u>2021/09/05 - Update arXiv'2104-2108</u>
- <u>2021/04/02 - Update CVPR 2021 and arXiv'2101-2103</u>
- <u>2021/01/10 - Update AAAI 2021 (Ada-Segment) and arXiv'2012</u>

## Panoptic Segmentation

### Tasks
[Panoptic segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf) unifies the typically distinct tasks of semantic segmentation (assign a class label to each pixel) and instance segmentation (detect and segment each object instance). Each pixel of an image must be assigned a semantic label and an instance id. Pixels with the same label and id belong to the same object; for stuff labels the instance id is ignored. 

![Panoptic segmentation](https://github.com/YvesXu/panoptic-segmentation-paper-list/blob/main/img/panoptic.png)

### Metrics

<a href="https://www.codecogs.com/eqnedit.php?latex=PQ&space;=&space;\frac{\sum_{(p,g)\in&space;TP}IoU(p,g)}{|TP|&plus;\frac{1}{2}|FP|&plus;\frac{1}{2}|FP|}&space;=&space;\underbrace{\frac{\sum_{(p,g)\in&space;TP}IoU(p,g)}{|TP|}}_{\text{segmentation&space;quality&space;(SQ)}}\times&space;\underbrace{\frac{|TP|}{|TP|&plus;\frac{1}{2}|FP|&plus;\frac{1}{2}|FP|}}_{\text{recognition&space;quality(RQ)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?PQ&space;=&space;\frac{\sum_{(p,g)\in&space;TP}IoU(p,g)}{|TP|&plus;\frac{1}{2}|FP|&plus;\frac{1}{2}|FP|}&space;=&space;\underbrace{\frac{\sum_{(p,g)\in&space;TP}IoU(p,g)}{|TP|}}_{\text{segmentation&space;quality&space;(SQ)}}\times&space;\underbrace{\frac{|TP|}{|TP|&plus;\frac{1}{2}|FP|&plus;\frac{1}{2}|FP|}}_{\text{recognition&space;quality(RQ)}}" title="PQ = \frac{\sum_{(p,g)\in TP}IoU(p,g)}{|TP|+\frac{1}{2}|FP|+\frac{1}{2}|FP|} = \underbrace{\frac{\sum_{(p,g)\in TP}IoU(p,g)}{|TP|}}_{\text{segmentation quality (SQ)}}\times \underbrace{\frac{|TP|}{|TP|+\frac{1}{2}|FP|+\frac{1}{2}|FP|}}_{\text{recognition quality(RQ)}}" /></a>


### Datesets
- [COCO](https://cocodataset.org/#panoptic-leaderboard)
- [Cityscapes](https://www.cityscapes-dataset.com/benchmarks/#panoptic-scene-labeling-task)
- [Mapillary](https://www.mapillary.com/dataset/challenge?pKey=rwbBtYKofke2NeLIvj8j-A)
- [Indian Driving](http://idd.insaan.iiit.ac.in/evaluation/an19-leader-board/)

## Paper List
### CVPR
#### 2021
- Cross-View Regularization for Domain Adaptive Panoptic Segmentation [[paper]](https://arxiv.org/pdf/2103.02584)
- Boundary IoU: Improving Object-Centric Image Segmentation Evaluation[[paper]](https://arxiv.org/pdf/2103.16562) [[project page]](https://bowenc0221.github.io/boundary-iou) [[code]](https://github.com/bowenc0221/boundary-iou-api)


#### 2020
- Bidirectional Graph Reasoning Network for Panoptic Segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Bidirectional_Graph_Reasoning_Network_for_Panoptic_Segmentation_CVPR_2020_paper.pdf)
- Pixel Consensus Voting for Panoptic Segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Pixel_Consensus_Voting_for_Panoptic_Segmentation_CVPR_2020_paper.pdf) [[empty code]](https://github.com/w-hc/pcv)
- BANet: Bidirectional Aggregation Network with Occlusion Handling for Panoptic Segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_BANet_Bidirectional_Aggregation_Network_With_Occlusion_Handling_for_Panoptic_Segmentation_CVPR_2020_paper.pdf) [[empty code]](https://github.com/Mooonside/BANet)
- Unifying Training and Inference for Panoptic Segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Unifying_Training_and_Inference_for_Panoptic_Segmentation_CVPR_2020_paper.pdf)
- Real-Time Panoptic Segmentation from Dense Detections [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hou_Real-Time_Panoptic_Segmentation_From_Dense_Detections_CVPR_2020_paper.pdf) [[code]](https://github.com/TRI-ML/realtime_panoptic)
- **[Panoptic-deeplab]** Panoptic-deeplab: A simple, strong, and fast baseline for bottom-up panoptic segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.pdf) [[code]](https://github.com/facebookresearch/detectron2) [[unofficial code]](https://github.com/bowenc0221/panoptic-deeplab)
- **[OCFusion]** Learning Instance Occlusion for Panoptic Segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lazarow_Learning_Instance_Occlusion_for_Panoptic_Segmentation_CVPR_2020_paper.pdf) [[code]](https://github.com/jlazarow/learning_instance_occlusion)


#### 2019
- Seamless Scene Segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Porzi_Seamless_Scene_Segmentation_CVPR_2019_paper.pdf)
- **[OANet]** An End-to-End Network for Panoptic Segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_An_End-To-End_Network_for_Panoptic_Segmentation_CVPR_2019_paper.pdf) [[blog]](https://yvesxu.com/2021/06/01/20210601-papers-005/)
- **[UPSNet]** UPSNet: A Unified Panoptic Segmentation Network [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xiong_UPSNet_A_Unified_Panoptic_Segmentation_Network_CVPR_2019_paper.pdf) [[code]](https://github.com/uber-research/UPSNet) [[blog]](https://yvesxu.com/2021/01/11/20210111-papers-004/)
- **[Panoptic FPN]** Panoptic Feature Pyramid Networks [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Feature_Pyramid_Networks_CVPR_2019_paper.pdf) [[code]](https://github.com/facebookresearch/detectron2) [[blog]](http://yvesxu.com/2020/11/11/20201111-papers-001/)
- **[<span id="aunet">AUNet</span>]** Attention-guided Unified Network for Panoptic Segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Attention-Guided_Unified_Network_for_Panoptic_Segmentation_CVPR_2019_paper.pdf) [[blog]](http://yvesxu.com/2020/12/01/20201201-papers-002/)
- Panoptic Segmentation [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf) 

### ECCV
#### 2020
- Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490103.pdf) [[unofficial code]](https://github.com/csrhddlam/axial-deeplab)

#### 2018
- Weakly- and Semi-Supervised Panoptic Segmentation [[paper]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Anurag_Arnab_Weakly-_and_Semi-Supervised_ECCV_2018_paper.pdf) [[code]](https://github.com/qizhuli/Weakly-Supervised-Panoptic-Segmentation)

### ICCV
#### 2019
- AdaptIS: Adaptive Instance Selection Network [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sofiiuk_AdaptIS_Adaptive_Instance_Selection_Network_ICCV_2019_paper.pdf) [[code]](https://github.com/saic-vul/adaptis/tree/pytorch)

### AAAI

#### 2021
- Ada-Segment: Automated Multi-Loss Adaptation for Panoptic Segmentation[[paper]](https://arxiv.org/pdf/2012.03603.pdf)

#### 2020
- SOGNet: Scene Overlap Graph Network for Panoptic Segmentation [[paper]](
https://ojs.aaai.org/index.php/AAAI/article/download/6955/6809) [[code]](https://github.com/LaoYang1994/SOGNet)

### IROS 
#### 2020
- Single-Shot Panoptic Segmentation [[paper]](https://ras.papercept.net/proceedings/IROS20/0102.pdf)

### NIPS
#### 2020
- Auto-Panoptic: Cooperative Multi-Component Architecture Search for Panoptic Segmentation [[paper]](https://papers.nips.cc/paper/2020/file/ec1f764517b7ffb52057af6df18142b7-Paper.pdf) [[code]](https://github.com/Jacobew/AutoPanoptic)
- Stable and expressive recurrent vision models [[paper]](https://papers.nips.cc/paper/2020/file/766d856ef1a6b02f93d894415e6bfa0e-Paper.pdf) [[empty code]](https://github.com/c-rbp/c-rbp)
- SOLOv2: Dynamic and Fast Instance Segmentation [[paper]](https://papers.nips.cc/paper/2020/file/cd3afef9b8b89558cd56638c3631868a-Paper.pdf) [[code]](https://github.com/aim-uofa/AdelaiDet/)

### Others
- Single Network Panoptic Segmentation for Street Scene Understanding | IV'19 | [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8813788) [[code]](https://github.com/DdeGeus/single-network-panoptic-segmentation)
- Fast Panoptic Segmentation Network | RA-L'19 | [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8972471)
- SpatialFlow: Bridging All Tasks for Panoptic Segmentation | TCSVT'20 |[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9180262) [[code]](https://github.com/chensnathan/SpatialFlow)

### arXiv
#### 2021
- Fully Convolutional Networks for Panoptic Segmentation with Point-based Supervision | 2108 | [[paper]](https://arxiv.org/pdf/2108.07682.pdf)
- Per-Pixel Classification is Not All You Need for Semantic Segmentation | 2107 |[[paper]](https://arxiv.org/pdf/2107.06278.pdf)
- SE-PSNet: Silhouette-based Enhancement Feature for Panoptic Segmentation Network | 2107 |[[paper]](https://arxiv.org/pdf/2107.05093.pdf)
- K-Net: Towards Unified Image Segmentation | 2107 | [[paper]](https://arxiv.org/pdf/2106.14855.pdf)
- Hierarchical Lovasz Embeddings for Proposal-free Panoptic Segmentation | 2106 | [[paper]](https://arxiv.org/pdf/2106.04555.pdf)
- Combinatorial Optimization for Panoptic Segmentation: An End-to-End Trainable Approach | 2106 | [[paper]](https://arxiv.org/pdf/2106.03188.pdf)
- Multi-Scale Feature Aggregation by Cross-Scale Pixel-to-Region Relation Operation for Semantic Segmentation | 2106 | [[paper]](https://arxiv.org/pdf/2106.01744.pdf)
- Exemplar-Based Open-Set Panoptic Segmentation Network | 2105 | [[paper]](https://arxiv.org/pdf/2105.08336.pdf)
- Boundary IoU: Improving Object-Centric Image Segmentation Evaluation | 2103 | [[paper]](https://arxiv.org/pdf/2103.16562.pdf)
- Panoptic Lintention Network: Towards Efficient Navigational Perception for the Visually Impaired | 2103 | [[paper]](https://arxiv.org/abs/2103.04128)
- Panoramic Panoptic Segmentation: Towards Complete Surrounding Understanding via Unsupervised Contrastive Learning | 2103 | [[paper]](https://arxiv.org/abs/2103.00868) [[empty code]](https://github.com/alexanderjaus/PPS)
- Instance and Panoptic Segmentation Using Conditional Convolutions | 2102 | [[paper]] (https://arxiv.org/pdf/2102.03026.pdf) [[code]](https://github.com/CurryYuan/InstanceRefer) [[code]](https://github.com/aim-uofa/AdelaiDet/)
- Joint Forecasting of Features and Feature Motion for Dense Semantic Future Prediction | 2101 | [[paper]](https://arxiv.org/pdf/2101.10620.pdf)

#### 2020
- Improving Panoptic Segmentation at All Scales [[paper]](https://arxiv.org/pdf/2012.07717.pdf)
- ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic Segmentation [[paper]](https://arxiv.org/pdf/2012.05258.pdf) [[demo]](https://youtu.be/XR4HFiwwao0)  [[code]](https://github.com/joe-siyuan-qiao/ViP-DeepLab)
- Single-shot Path Integrated Panoptic Segmentation | 2012 | [[paper]](https://arxiv.org/pdf/2012.01632.pdf)
- MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers | 2012 | [[paper]](https://arxiv.org/pdf/2012.00759.pdf)
- Fully Convolutional Networks for Panoptic Segmentation | 2012 | [[paper]](https://arxiv.org/pdf/2012.00720.pdf) [[code]](https://github.com/Jia-Research-Lab/PanopticFCN)
- Scaling Wide Residual Networks for Panoptic Segmentation | 2011 | [[paper]](https://arxiv.org/pdf/2011.11675.pdf)
- Unifying Instance and Panoptic Segmentation with Dynamic Rank-1 Convolutions | 2011 | [[paper]](https://arxiv.org/pdf/2011.09796.pdf)
- Learning Panoptic Segmentation from Instance Contours | 2010 | [[paper]](https://arxiv.org/pdf/2010.11681.pdf)
- Learning Category- and Instance-Aware Pixel Embedding for Fast Panoptic Segmentation | 2009 | [[paper]](https://arxiv.org/pdf/2009.13342.pdf)
- CASNet: Common Attribute Support Network for image instance and panoptic segmentation | 2008 | [[paper]](https://arxiv.org/pdf/2008.00810.pdf)
- PanoNet: Real-time Panoptic Segmentation through Position-Sensitive Feature Embedding | 2008 | [[paper]](https://arxiv.org/pdf/2008.00192.pdf)
- DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution | 2006 | [[paper]](https://arxiv.org/pdf/2006.02334.pdf)
- End-to-End Object Detection with Transformers | 2005 | [[paper]](https://arxiv.org/pdf/2005.12872.pdf)
- EfficientPS: Efficient Panoptic Segmentation | 2004 | [[paper]](https://arxiv.org/pdf/2004.02307.pdf)
- EPSNet: Efficient Panoptic Segmentation Network with Cross-layer Attention Fusion | 2003 |[[paper]](https://arxiv.org/pdf/2003.10142.pdf)
- Towards Bounding-Box Free Panoptic Segmentation | 2002 | [[paper]](https://arxiv.org/pdf/2002.07705.pdf)

#### 2019
- Bipartite Conditional Random Fields for Panoptic Segmentation | 1912 | [[paper]](https://arxiv.org/pdf/1912.05307.pdf)
- Generator evaluator-selector net for panoptic image segmentation and splitting unfamiliar objects into parts | 1908 |  [[paper]](https://arxiv.org/ftp/arxiv/papers/1908/1908.09108.pdf)
#### 2018
- Learning to Fuse Things and Stuff | 1812 | [[paper]](https://arxiv.org/pdf/1812.01192.pdf)
- Panoptic segmentation with a joint semantic and instance segmentation network | 1809 | [[paper]](https://arxiv.org/abs/1809.02110)

### Related Work
#### LiDAR Panoptic Segmentation
- SMAC-Seg: LiDAR Panoptic Segmentation via Sparse Multi-directional Attention Clustering | arXiv'21 | [[paper]](https://arxiv.org/pdf/2108.13588.pdf)
- A Technical Survey and Evaluation of Traditional Point Cloud Clustering Methods for LiDAR Panoptic Segmentation | arXiv'21 | [[paper]](https://arxiv.org/pdf/2108.09522.pdf)
- GP-S3Net: Graph-based Panoptic Sparse Semantic Segmentation Network | arXiv'21 | [[paper]](https://arxiv.org/pdf/2108.08401.pdf)
- Panoptic-PolarNet: Proposal-free LiDAR Point Cloud Panoptic Segmentation | CVPR'21 | [[paper]](https://arxiv.org/pdf/2103.14962.pdf) [[code]](https://github.com/edwardzhou130/Panoptic-PolarNet)
- 4D Panoptic LiDAR Segmentation | arXiv'21 | [[paper]](https://arxiv.org/pdf/2102.12472)
- LiDAR Panoptic Segmentation for Autonomous Driving | IROS'20 | [[paper]](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2020iros.pdf)
- EfficientLPS: Efficient LiDAR Panoptic Segmentation | arXiv'21 | [[paper]](https://arxiv.org/pdf/2102.08009.pdf)
- Panoster: End-to-end Panoptic Segmentation of LiDAR Point Clouds | arXiv'20 | [[paper]](https://arxiv.org/pdf/2010.15157.pdf)

#### Video Panoptic Segmetation
- Merging Tasks for Video Panoptic Segmentation | arXiv'21 | [[paper]](https://arxiv.org/pdf/2108.04223.pdf)
- Learning to Associate Every Segment for Video Panoptic Segmentation | arXiv'21 | [[paper]](https://arxiv.org/pdf/2106.09453.pdf)
- Panoptic Segmentation Forecasting | arXiv'21 | [[paper]](https://arxiv.org/pdf/2104.03962.pdf)
- Video Panoptic Segmentation | CVPR'20 | [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Video_Panoptic_Segmentation_CVPR_2020_paper.pdf)

#### Others
- Bird's-Eye-View Panoptic Segmentation Using Monocular Frontal View Images | arXiv'21 | [[paper]](https://arxiv.org/pdf/2108.03227.pdf)
- Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks | arXiv'21 | [[paper]](https://arxiv.org/pdf/2107.07933.pdf)
- Part-aware Panoptic Segmentation | arXiv'21 | [[paper]](https://arxiv.org/pdf/2106.06351.pdf)


- SceneGraphFusion: Incremental 3D Scene Graph Prediction from RGB-D Sequences | CVPR'21 | [[paper]](https://arxiv.org/pdf/2103.14898.pdf) [[project page]](https://shunchengwu.github.io/SceneGraphFusion)
- Video Class Agnostic Segmentation Benchmark for Autonomous Driving | arXiv'21 | [[paper]](https://arxiv.org/pdf/2103.11015.pdf)
- InstanceRefer: Cooperative Holistic Understanding for Visual Grounding on Point Clouds through Instance Multi-level Contextual Referring | arXiv'21 | [[paper]](https://arxiv.org/pdf/2103.01128.pdf) [[code]](https://github.com/CurryYuan/InstanceRefer)
- STEP: Segmenting and Tracking Every Pixel | arXiv'21 | [[paper]](https://arxiv.org/pdf/2102.11859.pdf)
- Adversarial Segmentation Loss for Sketch Colorization | arXiv'21 | [[paper]](https://arxiv.org/pdf/2102.06192.pdf) [[code]](https://github.com/giddyyupp/AdvSegLoss)
- Can we cover navigational perception needs of the visually impaired by panoptic segmentation? | arXiv'20 | [[paper]](https://arxiv.org/pdf/2007.10202.pdf)
- Efficient Full Image Interactive Segmentation by Leveraging Within-image Appearance Similarity | arXiv'20 | [[paper]](https://arxiv.org/pdf/2007.08173.pdf)
- Panoptic Instance Segmentation on Pigs | arXiv'20 |[[paper]](https://arxiv.org/pdf/2005.10499.pdf)
- Naive-Student: Leveraging Semi-Supervised Learning in Video Sequences for Urban Scene Segmentation | ECCV'20 | [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540664.pdf)
- Unsupervised Instance Segmentation in Microscopy Images via Panoptic Domain Adaptation and Task Re-Weighting | CVPR'20 | [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Unsupervised_Instance_Segmentation_in_Microscopy_Images_via_Panoptic_Domain_Adaptation_CVPR_2020_paper.pdf)
- Panoptic-Based Image Synthesis | CVPR'20 | [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dundar_Panoptic-Based_Image_Synthesis_CVPR_2020_paper.pdf)
- MOPT: Multi-Object Panoptic Tracking | arXiv'20 | [[paper]](https://arxiv.org/pdf/2004.08189)
- Panoptic Segmentation-Based Attention for Image Captioning | Applied Sciences'20 | [[paper]](https://www.mdpi.com/2076-3417/10/1/391/htm)
- PanDA: Panoptic Data Augmentation | arXiv'19 | [[paper]](https://arxiv.org/pdf/1911.12317.pdf)
- Panoptic Image Annotation with a Collaborative Assistant | MM'20 | [[paper]](https://arxiv.org/pdf/1906.06798.pdf)
- IMP: Instance Mask Projection for High Accuracy Semantic Segmentation of Things | ICCV'19 | [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_IMP_Instance_Mask_Projection_for_High_Accuracy_Semantic_Segmentation_of_ICCV_2019_paper.pdf)
- Panoptic Edge Detection | arXiv'19 | [[paper]](https://arxiv.org/pdf/1906.00590.pdf)
- Detecting Reflections by Combining Semantic and Instance Segmentation | arXiv'19 | [[paper]](https://arxiv.org/pdf/1904.13273)
- Panopticfusion: Online volumetric semantic mapping at the level of stuff and things | IROS'19 | [[paper]](https://arxiv.org/pdf/1903.01177.pdf)
- Interactive Full Image Segmentation by Considering All Regions Jointly | CVPR'19 | [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Agustsson_Interactive_Full_Image_Segmentation_by_Considering_All_Regions_Jointly_CVPR_2019_paper.pdf)


## Benchmarks

### Cityscapes
#### *val* set
|title|PQ|PQ<sup>Th</sup>|PQ<sup>St</sup>|
|-|-|-|-|
|OCFusion(ResNet-50)|60.2|54.0|64.7|
|Real-Time Panoptic Segmentation(ResNet-50)|58.8|52.1|63.7|
|AUNet(ResNet-50)|56.4|52.7|59.0|
|AUNet(ResNet-101)|59.0|54.8|62.1|


#### *test* set
|title|PQ|PQ<sup>Th</sup>|PQ<sup>St</sup>|
|-|-|-|-|
|Panoptic-DeepLab |65.5|||

### COCO 
#### *val* set
|title|PQ|PQ<sup>Th</sup>|PQ<sup>St</sup>|
|-|-|-|-|
|OCFusion(ResNeXt-101)|46.3|53.5|35.4|
|Panoptic FPN|58.1|52.0|62.5|
|Panoptic-DeepLab(Xception-71)|41.2|44.9|35.7|
|Real-Time Panoptic Segmentation(ResNet-50)|37.1|41.0|31.3|
|OANet(ResNet-50)|39.0|48.3|24.9|
|OANet(ResNet-101)|40.7|50.0|26.6|

#### *test-dev* set
|title|year|PQ|PQ<sup>Th</sup>|PQ<sup>St</sup>|
|-|-|-|-|-|
|Panoptic FPN||40.9|48.3|29.7|
|AUNet(ResNet-101)||45.2|54.4|31.3|
|AUNet(ResNet-152)||45.5|54.7|31.6|
|AUNet(ResNeXt-152)||46.5|55.8|32.5|
|OCFusion(ResNeXt-101)||46.7|54.0|35.7|
|Panoptic-DeepLab(Xception-71)||41.4|45.1|35.9|
|OANet(ResNet-101)|2018|41.3|50.4|27.7|

### Mapillary Vistas
#### *val* set
|title|PQ|PQ<sup>Th</sup>|PQ<sup>St</sup>|
|-|-|-|-|
|OCFusion|-|-|-|
|Panoptic-DeepLab(Xception-71)|40.3|33.5|49.3|

#### *test* set
|title|PQ|PQ<sup>Th</sup>|PQ<sup>St</sup>|
|-|-|-|-|
|OCFusion|-|-|-|
|Panoptic-DeepLab(Xception-71)|42.7|35.9|51.6|
