# FH-Seg
Accurate fine-grained segmentation of the renal vasculature is critical for nephrological analysis, yet it faces challenges due to diverse and insufficiently annotated images. Existing methods struggle to accurately segment intricate regions of the renal vasculature, such as the inner and outer walls, arteries and lesions. In this paper, we introduce FH-Seg, a Full-scale Hierarchical Learning Framework designed for comprehensive segmentation of the renal vasculature. Specifically, FH-Seg employs full-scale skip connections that merge detailed anatomical information with contextual semantics across scales, effectively bridging the gap between structural and pathological contexts. Additionally, we implement a learnable hierarchical soft attention gates to adaptively reduce interference from non-core information, enhancing the focus on critical vascular features. To advance research on renal pathology segmentation, we also developed a Large Renal Vasculature (LRV) dataset, which contains 16,214 fine-grained annotated images of 5,600 renal arteries. Extensive experiments on the LRV dataset demonstrate FH-Seg’s superior accuracies (71.23\% Dice, 73.06\% F1), outperforming Omni-Seg by 2.67 and 2.13 percentage points respectively. 

### Contents
1. [Requirements](#Requirements)
2. [Installation](#Installation)
3. [Demo](#Demo)
4. [Experiments](#Experiments)


### Requirements
- Python 3.7.0
- PyTorch-0.4.1
- CUDA Version 11.1

### Installation
- Install Python 3.7.0
- pip install -r requirements.txt


### Demo

- After successfully completing basic installation, you'll be ready to run the demo.
1. Clone the FH-Seg repository

2. Run the training code. 
```
python train_2D_patch_scale_aug_FH_Seg_for_6class_1GPU.py  # single-gpu training on gpu 0
```

3. Run the test code
```
python Validation_FH_Seg_for_6_class_1GPU_2k.py
```

### Experiments
- Computational Cost 
```
GPU：NVIDIA Tesla P40
```
=======
# FH-Seg
The official code for FH-Seg
>>>>>>> ff4fc77a8c56c6eaaffd281ed94617f41d574e90
