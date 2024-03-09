# Pseudo-Label The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks

The repository implement a semi-supervised method for Deep Neural Networks, the Pseudo Label. More details for the method please refer to *Pseudo-Label The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks*.
## my understanding
- 在进行数据加载(dataloader)的时候,一个批次(batch)里面包含了有标签数据和无标签数据。
- “for batch_idx, (data, targets) in enumerate(data_loader)”中有标签数据的标签即为真实标签，无标签数据的标签即为-1(NO_LABLE)。
- 每个批次放入模型训练后，会进行两部分的损失计算，一部分是有标签数据的损失,一部分是无标签数据的损失。最后两部分的损失加起来即为总损失
## The environment:

- Python 3.6.5 :: Anaconda
- PyTorch 0.4.0
- torchvision 0.2.1
- tensorboardX (for log)
- tensorflow (for visualization)

## To prepare the data:
```shell
bash data-local/bin/prepare_cifar10.sh
```

## To run the code:
```shell
python -m experiments.cifar10_test
```

## Visualization:
Make sure you have installed the tensorflow for tensorboard
```shell
tensorboard --logdir runs
```


## Code Reference

[pytorch-cifar@kuangliu](https://github.com/kuangliu/pytorch-cifar)

[mean-teacher@CuriousAI](https://github.com/CuriousAI/mean-teacher)

[senet.pytorch@moskomule](https://github.com/moskomule/senet.pytorch)

