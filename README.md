# Face-Parsing-Network
PyTorch implementation of Face Parsing (based on semantic segmentation)

Network originally based on :
[Object contour detection with a fully convolutional encoder-decoder network [J. Yang, 2016]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_Object_Contour_Detection_CVPR_2016_paper.pdf)

Used for :
[Generative Face Completion [Y. Li, 2017]](https://arxiv.org/abs/1704.05838)

Used Dataset :
[CelebA-HQ Dataset](https://github.com/switchablenorms/CelebAMask-HQ)

The network layer codes(like batch normalization, convolution...) was refer to [Context Attention](https://github.com/DAA233/generative-inpainting-pytorch)

-------------------------
+ Dataset and ground-truth should be located as:
  - Training image directory
    + 1.jpg
    + 2.jpg
  - Ground-truth image directory
    + 1.png
    + 2.png



-------------------------
+ Trained PyTorch model from CelebA-HQ dataset : [Here](https://drive.google.com/file/d/1Rx9R-HOax7-Y3C4lwR_KIi_HdEYSfqkQ/view?usp=share_link)
+ Samples of training and ground-truth set : [Download](https://drive.google.com/file/d/1-q1s_4OU9QzLHy_zk-V53DpiIaMlPeVy/view?usp=sharing)

Quick inference using pre-trained model
-----------
```
$ cp parser_00100000.pt checkpoints/
$ python seg_inference.py
```

Then samples are saved in output directory

----------

<samples>
<img src="https://user-images.githubusercontent.com/10590942/69032733-88e28800-0a20-11ea-8712-014a49eb3458.png" width="90%"></img>
Trained with 30,000 CelebA-HQ dataset and applied it to LFW and CelebA dataset using that pre-trained model

PyTorch 1.1 + Python3.6
