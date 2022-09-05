# AIParsing
This respository includes a PyTorch implementation of the TIP2022 paper AIParsing:Anchor-Free Instance-Level Human Parsing. 

## Requirements:<br>
python 3.7<br>
PyTorch 1.7.1<br>
cuda 10.1 <br>

## Compiling<br>

cd models<br>
sh make.sh<br>

cd ..<br>
python build.py build develop<br>


## Dataset and pretrained model<br>
Plesae download [CIHP](https://drive.google.com/open?id=1OLBd23ufm6CU8CZmLEYMdF-x2b8mRgxV) dataset<br>

[pretrained model](https://##)<br>


## Evaluation<br>
bash test_CIHP_R50_75epoch.sh<br>

## Training<br>
bash train_CIHP_R50_75epoch.sh<br>

## Acknowledgment  
This project is created based on the [Parsing R-CNN](https://github.com/soeaver/Parsing-R-CNN), [CenterMask](https://github.com/youngwanLEE/CenterMask)

If this code is helpful for your research, please cite the following paper:

<p>
@article{AIParsing2022,<br>
     title={AIParsing: Anchor-Free Instance-Level Human Parsing},<br>
     author={Sanyi Zhang, Xiaochun Cao, Guo-jun Qi, Zhanjie Song, Jie Zhou},<br>
     journal={IEEE Transactions on Image Processing (TIP)},<br>
     year={2022},<br>
     volume={31},<br>
     pages={5599-5612}<br> 
}
  </p>
