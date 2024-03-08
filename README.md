# Texture Correlation Network
The source code for our paper "Pose-Guided Generation of Human lmages via Texture Transfer."

## Get Start
### 1) Requirement

* Python 3.7.9
* Pytorch 1.7.1
* torchvision 0.8.2
* CUDA 11.1

### 2) Data Preperation

Following **[PATN](https://github.com/tengteng95/Pose-Transfer)**, the dataset split files and extracted keypoints files can be obtained as follows:

**DeepFashion**


* Download the DeepFashion dataset **[in-shop clothes retrival benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)**, and put them under the `./Fashion` directory.

* Download train/test pairs and train/test keypoints annotations from **[Google Drive](https://drive.google.com/drive/folders/1qZDod3QDD7PaBxnNyHCuLBR7ftTSkSE1?usp=sharing)**, including **fasion-resize-pairs-train.csv, fasion-resize-pairs-test.csv, fasion-resize-annotation-train.csv, fasion-resize-annotation-train.csv, train.lst, test.lst**, and put them under the `./Fashion` directory.

* Split the raw image into the training set (`./Fashion/train`) and test set (`./Fashion/test`):
``` bash
python data/generate_fashion_datasets.py
```

**Market1501**

* Download the Market1501 dataset from **[here](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)**. Rename **bounding_box_train** and **bounding_box_test** as **train** and **test**, and put them under the `./Market` directory.

* Download train/test key points annotations from **[Google Drive](https://drive.google.com/drive/folders/1zzkimhX_D5gR1G8txTQkPXwdZPRcnrAx?usp=sharing)** including **market-pairs-train.csv, market-pairs-test.csv, market-annotation-train.csv, market-annotation-train.csv**. Put these files under the `./Market` directory.

### 3) Train
#### DeepFashion
Stage1:
``` 
python train.py --name HPT_fashion --model Coarse_Image --checkpoints_dir ./checkpoints --dataset_mode fashion --dataroot Fashion --batchSize 16
```
Stage2:
``` 
python train.py --name HPT_fashion --model Texture_Transfer --checkpoints_dir ./checkpoints --dataset_mode fashion --dataroot Fashion --lambda_style 1000 --lambda_content 1 --lambda_lg1 2 --lambda_lg2 5 --lambda_g 2 --batchSize 2 --continue_train
``` 

#### Market
``` 
python train.py --name HPT_market --model Texture_Transfer --checkpoints_dir ./checkpoints --dataset_mode market --dataroot XXXX/Market --lambda_style 500 --lambda_content 0.5 --lambda_lg1 1 --lambda_lg2 2 --lambda_g 5 --batchSize 64
``` 

### 4) Test
We will provide our training results at https://drive.google.com/drive/folders/1Fk31hPdtLHpvQoVAi7OtkEmdx-gh1iCY?usp=drive_link
#### DeepFashion
``` 
python test.py --name HPT_fashion --model Texture_Transfer --checkpoints_dir ./checkpoints --dataset_mode fashion --dataroot Fashion
``` 
#### Market
``` 
python test.py --name HPT_market --model Texture_Transfer --checkpoints_dir ./checkpoints --dataset_mode market --dataroot Market 
``` 

### 5) Evaluation

We adopt SSIM, PSNR, FID, LPIPS and person re-identification (re-id) system for the evaluation. Please clone the official repository **[PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity/tree/future)** of the LPIPS score, and put the folder PerceptualSimilarity to the folder **[metrics](https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network/tree/main/metrics)**.

* For SSIM, PSNR, FID and LPIPS:(--distorated_path: folder containing the images obtained after your training. )

**DeepFashion**
``` bash
python -m  metrics.metrics --gt_path=./Fashion/test --distorated_path=./results --fid_real_path=./Fashion/test --name=./fashion
``` 

## Acknowledgement 

We build our project base on "Lightweight Texture Correlation Network for Pose Guided Person Image Generation". Some dataset preprocessing methods are derived from (https://github.com/tengteng95/Pose-Transfer).

