# DSFD（Dual Shot Face Detector）
源腾讯优图开源工程：[DSFD: Dual Shot Face Detector](https://github.com/TencentYoutuResearch/FaceDetection-DSFD)，极致精度的人脸检测算法。这是在腾讯Inference开源基础上，边学习边重构的预处理、数据加载、训练、评估测试、预测（视频帧逐帧检测）整个流程。
## Requirements
* PyTorch == 1.0.0   
* Torchvision == 0.4.0   
* Python == 3.6   
* NVIDIA GPU == Geforce 1080(8G)   
* Linux CUDA CuDNN   
## Getting Started
### Inference
根据[Demo](/demo.py)中单张img预测需求或者video预测需求，指明（注释）IMG_PATH or VIDEO_PATH：
```
python demo.py
```
### Training
首先准备WIDERFace数据集的预处理，解析'wider_face_train_bbx_gt.txt'与'wider_face_val_bbx_gt.txt'两个文件的标注信息：
```
python ./utils/my_data_preprocess.py
```
将生成的./data/train.txt与./data/val.txt文件，用于Dataloader。然后，可直接跑训练：
```
python train.py
```
## Result

