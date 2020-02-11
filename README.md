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
+ Trained PyTorch model using all 30,000 images from CelebA-HQ dataset : [Here](https://drive.google.com/open?id=1e38G_bTvsktDkgZRyG-V7Yk6gR9yO7u3)
(This pre-trained model was for making ground-truth for other face dataset)
+ [Here to download](https://drive.google.com/open?id=1oR4Ja2rO9k66zV8JTLtdOVrCnW7zH0xW) 30,000 CelebA-HQ images including parsing ground-truth images

<samples>
<img src="https://user-images.githubusercontent.com/10590942/69032733-88e28800-0a20-11ea-8712-014a49eb3458.png" width="90%"></img>
Trained with 30,000 CelebA-HQ dataset and applied it to LFW and CelebA dataset using that pre-trained model

PyTorch 1.1 + Python3.6
