
<div style="display: flex; justify-content: center; margin-bottom: 10px;">
    <img src="Framework_Architecture.png" style="width: 100vw; height: auto;">
</div>


<h1 align="left">
   MedMask: A Self-Supervised Masking Framework for Medical Image Detection Using VFMs.
</h1>


## What is MedMask?
MedMask is a self-supervised masking framework designed to enhance medical image detection, particularly for breast cancer detection from mammograms (BCDM). It addresses the challenge of limited annotated datasets by leveraging masked autoencoders (MAE) and vision foundation models (VFMs) within a transformer-based architecture. The framework employs a customized MAE module that masks and reconstructs multi-scale feature maps, allowing the model to learn domain-specific characteristics more effectively. Additionally, MedMask integrates an expert contrastive knowledge distillation technique, utilizing the zero-shot capabilities of VFMs to improve feature representations. By combining self-supervised learning with knowledge distillation, MedMask achieves state-of-the-art performance on publicly available mammogram datasets like INBreast and DDSM, demonstrating significant improvements in sensitivity. Its applicability extends beyond medical imaging, showcasing generalizability to natural image tasks.


## What is RSBA-BSD1K Data?

RSNA-BSD1K is a bounding box annotated subset of 1,000 mammograms from the RSNA Breast Screening Dataset, designed to support further research in breast cancer detection from mammograms (BCDM). The original RSNA dataset consists of 54,706 screening mammograms, containing 1,000 malignancies from 8,000 patients. From this, we curated RSNA-BSD1K, which includes 1,000 mammograms with 200 malignant cases, annotated at the bounding box level by two expert radiologists.



## 1. Installation

### 1.1 Requirements

- Linux, CUDA >= 11.1, GCC >= 8.4

- Python >= 3.8

- torch >= 1.10.1, torchvision >= 0.11.2

- Other requirements

  ```bash
  pip install -r requirements.txt
  ```

### 1.2 Compiling Deformable DETR CUDA operators

```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```



## 2. Usage

### 2.1 Data preparation

We provide the three benchmark datasets used in our experiments:

- **BCD_DDSM**: The DDSM dataset consists of 1,324 mammography images, including 573 malignant cases.  
- **BCD_InBreast**: The INBreast dataset contains 200 images from 115 patients, with 41 malignant and 159 benign cases.  
- **BCD_RSNA**: The RSNA-BSD1K dataset is a curated subset of 1,000 mammograms from the original RSNA dataset, including 200 malignant cases with bounding box annotations.


You can download the raw data from the official websites: [dataset]() and organize the datasets and annotations as follows:

```bash
[data_root]
└─ inbreast
	└─ annotations
		└─ instances_train.json
		└─ instances_val.json
	└─ images
		└─ train
		└─ val
└─ ddsm
	└─ annotations
		└─ instances_train.json
		└─ instances_val.json

	└─ images
		└─ train
		└─ val
└─ rsna-bsd1k
	└─ annotations
		└─ instances_full.json
		└─ instances_val.json
	└─ images
		└─ train
		└─ val
└─ cityscapes
	└─ annotations
		└─ cityscapes_train_cocostyle.json
		└─ cityscapes_train_caronly_cocostyle.json
		└─ cityscapes_val_cocostyle.json
		└─ cityscapes_val_caronly_cocostyle.json
	└─ leftImg8bit
		└─ train
		└─ val
└─ foggy_cityscapes
	└─ annotations
		└─ foggy_cityscapes_train_cocostyle.json
		└─ foggy_cityscapes_val_cocostyle.json
	└─ leftImg8bit_foggy
		└─ train
		└─ val
└─ sim10k
	└─ annotations
		└─ sim10k_train_cocostyle.json
		└─ sim10k_val_cocostyle.json
	└─ JPEGImages
└─ bdd10k
	└─ annotations
		└─ bdd100k_daytime_train_cocostyle.json
		└─ bdd100k_daytime_val_cocostyle.json
	└─ JPEGImages
```

To use additional datasets, you can edit [datasets/coco_style_dataset.py](https://github.com/JeremyZhao1998/MRT-release/blob/main/datasets/coco_style_dataset.py) and add key-value pairs to `CocoStyleDataset.img_dirs` and `CocoStyleDataset.anno_files` .


## 2.2 Training and Evaluation

Our method follows a three-stage training paradigm to optimize efficiency and performance. The initial stage involves standard supervised training using annotated mammograms. The second stage incorporates self-supervised masked autoencoding (MAE) to refine representations from unannotated data. The final stage introduces Expert-Guided Fine-Tuning, leveraging vision foundation models (VFMs) for enhanced feature learning and generalization.

For training on the DDSM-to-INBreast benchmark, update the files in `configs/medmask/ddsm/` to specify `DATA_ROOT` and `OUTPUT_DIR`, then execute:

```sh
sh configs/medmask/ddsm/source_only.sh      # Stage 1: Baseline Object Detector Training
sh configs/medmask/ddsm/cross_domain.sh     # Stage 2: Masked Autoencoder Training
```
## 3. Results and Model Parameters

## 4. Citation



Thanks for your attention.
