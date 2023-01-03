
## **Dynamic Perfusion Representation and Aggregation Network for Nodule Segmentation using Contrast-enhanced US**
Peng Wan, Haiyan Xue, Chunrui Liu, Fang Chen, Wentao Kong, Daoqiang Zhang

This repository contains the  Pytorch implementation of training & evaluation code of DpRAN model.

**DpRAN** is an automatic lesion segmentation method for dynamic contrast-enhanced US imaging, as shown in Figure 1.

<img src="https://github.com/wanpeng16/DpRAN/blob/main/framework.jpg"  style="zoom: 75%;" />
<p align="left">

**Figure 1**: The framework of dynamic perfusion representation and aggregation network (DpRAN). The proposed DpRAN method
decomposes the segmentation task into the two stages. With the identified critical point t’, the specifically designed perfusion fusion module is used to 
aggregate the dual-view dynamic perfusion representations along the up-sampling path.
</p>

## Requirements

For install and data preparation, please refer to the guidelines in [requirement.txt](./requirements.txt).

Python >=3.8 | Pytorch >=1.10.2 | CUDA >=11.3
```
pip install segmentation-models-pytorch==0.8.2
pip install torchmetrics==0.11.0
pip install segmentation-models-pytorch==0.3.1
pip install opencv-python==4.5.3.56
pip install spatial-correlation-sampler==0.3.0
pip install numpy==1.23.0
pip install scipy==1.6.3
```

## Data Preprocessing

- Run [extractframes.py](./data/preprocessing/extractframes.py) for temporal redundancy removal of CEUS videos.
- Customized dataloader for dynamic CEUS imaging [dataloader.py](./data/dataloader.py)
<br>
<img src="https://github.com/wanpeng16/DpRAN/blob/main/vis/11.jpg"  style="zoom: 50%;" />
<p align="=">

**Figure 2**: An illustration of the dynamic enhancement process of thyroid nodule.
</p>
## Evaluation
```
python inference.py
```

## Training
```
python main.py
```
## Visualization

<img src="https://github.com/wanpeng16/DpRAN/blob/main/vis/28.png"  style="zoom: 50%;" />
<br>
<img src="https://github.com/wanpeng16/DpRAN/blob/main/vis/165.png"  style="zoom: 50%;" />

<p align="=">

**Figure 3**: Visualization results of our DpRAN method. The first row is the three representative perfusion points which cover the initial enhancement, reaching peak, and the eventual wash out, as well as the ground truth. Segmentation results are overlaid on the peaking frame. The second row is the intermediate fusion features generated by the last CTA module and the restored map before the final convolution layer.
</p>

