# AIParsing
This respository includes a PyTorch implementation of the TIP2022 paper AIParsing:Anchor-Free Instance-Level Human Parsing. 

## Requirements:<br>
python 3.7<br>
PyTorch 1.7.1<br>
cuda 10.1 <br>

The detail environment can be find in AIParsing_env.yaml.

## Compiling<br>

Apex install:

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
or python setup.py build develop

cd models<br>
sh make.sh<br>

cd ..<br>
python setup.py build develop<br>


## Dataset and pretrained model<br>
Plesae download [CIHP](https://drive.google.com/open?id=1OLBd23ufm6CU8CZmLEYMdF-x2b8mRgxV) dataset<br>

[Well-trained models on the CIHP and LV-MHP datasets](https://tjueducn-my.sharepoint.com/:f:/g/personal/zhangsanyi_tju_edu_cn/EhLaPw8f2nBAsosG1FaOZ4MBkmcQkTy61SrvbOZ7jR9xHA?e=wbbmgq) (MM:aiparsing)<br>


## Evaluation<br>
bash test_CIHP_R50_75epoch.sh<br>

## Training<br>
bash train_CIHP_R50_75epoch.sh<br>

## Acknowledgment  
This project is created based on the [Parsing R-CNN](https://github.com/soeaver/Parsing-R-CNN), [CenterMask](https://github.com/youngwanLEE/CenterMask)

If this code is helpful for your research, please cite the following paper:

<p>
@article{AIParsing2022,<br>
&emsp;  title={AIParsing: Anchor-Free Instance-Level Human Parsing},<br>
&emsp;   author={Sanyi Zhang, Xiaochun Cao, Guo-jun Qi, Zhanjie Song, Jie Zhou},<br>
&emsp;       journal={IEEE Transactions on Image Processing (TIP)},<br>
&emsp;       year={2022},<br>
&emsp;       volume={31},<br>
&emsp;       pages={5599-5612}<br> 
}
  </p>
