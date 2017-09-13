# CycleGAN-TensorFlow
An implementation of CycleGan using TensorFlow (work in progress).

## References

* CycleGAN paper: https://arxiv.org/abs/1703.10593
* Official source code in Torch: https://github.com/junyanz/CycleGAN
* WGAN :https://github.com/igul222/improved_wgan_training

## 一些训练心得

* 使用sigmoid交叉熵尽管效果不尽如人意，但最容易训练成功，Wgan和最小二乘的训练方式都有很大几率训练失败。
* 训练集大小采用每个图集5W-10W张
* 初始化权重十分敏感，mean建议为0.0，非常容易在训练过程中图像全黑
* 图像预处理的uint8和float32要特别注意，非常容易导致训练图片失效
* loss为各个部分加和时要尽量让各个部分数量级一致，否则会训练失衡
* 学习速率要尽可能小2e-4是我的建议数值

![image](https://github.com/KumoKyaku/READMEPictures/blob/master/CycleGANTest/sigmoid_cross_entropy.jpg)

![image](https://github.com/KumoKyaku/READMEPictures/blob/master/CycleGANTest/sigmoid_cross_entropy_res.jpg)

## Others

![image](https://github.com/KumoKyaku/READMEPictures/blob/master/want/wanted.jpg)