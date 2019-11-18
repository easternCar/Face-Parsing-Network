# Face-Parsing-Network
PyTorch implementation of Face Parsing (based on semantic segmentation)

Network originally based on :
Object contour detection with a fully convolutional encoder-decoder network [J. Yang, 2016] http://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_Object_Contour_Detection_CVPR_2016_paper.pdf

Used for :
Generative Face Completion [Y. Li, 2017] https://arxiv.org/abs/1704.05838

Used Dataset :
CelebA-HQ Dataset https://github.com/switchablenorms/CelebAMask-HQ

The network layer codes(like batch normalization, convolution...) was refer to https://github.com/DAA233/generative-inpainting-pytorch

(https://user-images.githubusercontent.com/10590942/69032733-88e28800-0a20-11ea-8712-014a49eb3458.png)
(This examples are outputs of LFW and CelebA using the model trained on CelebA-HQ dataset)
