#!coding:utf-8
import torch
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from pathlib import Path
from util.datasets import NO_LABEL

class PseudoLabel:

    def __init__(self, model, optimizer, loss_fn, device, config, writer=None, save_dir=None, save_freq=5):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.device = device
        self.writer = writer
        self.labeled_bs = config.labeled_batch_size
        self.global_step = 0
        self.epoch = 0
        self.T1, self.T2 = config.t1, config.t2 # 't1': 100, 't2': 600,
        self.af = config.af
        
    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_loss = [] # loop_loss用于记录每个迭代的损失
        accuracy = [] # accuracy用于记录每个迭代的准确率
        labeled_n = 0 # 用于记录标记样本的数量
        mode = "train" if is_train else "test" # 用于表示当前是训练模式还是测试模式
        for batch_idx, (data, targets) in enumerate(data_loader): # 迭代数据加载器中的每个批次
            self.global_step += batch_idx
            data, targets = data.to(self.device), targets.to(self.device) # 计算标记样本的损失labeled_loss 感觉里面还是有无标签数据的data和targets的
            outputs = self.model(data) # 长度为128
            if is_train: # 如果是训练模式
                labeled_bs = self.labeled_bs # base_labeled_batch_size': 64 
                labeled_loss = torch.sum(self.loss_fn(outputs, targets)) / labeled_bs # 首先计算标记样本的损失labeled_loss 损失函数设置了忽略无标签的数据(main.py)
                with torch.no_grad(): # 避免对这部分计算进行梯度计算
                    pseudo_labeled = outputs.max(1)[1] # 使用无梯度计算获取伪标签pseudo_labeled
                unlabeled_loss = torch.sum(targets.eq(NO_LABEL).float() * self.loss_fn(outputs, pseudo_labeled)) / (data.size(0)-labeled_bs +1e-10)
                # 无标签数据的targets是“ NO_LABEL ”
                '''
                在计算无标签样本的损失时，将(data.size(0) - labeled_bs)作为分母时，可能会出现分母为零的情况。为了避免除以零的错误，通常会在分母上加上一个很小的常数（例如1e-10），以确保分母不为零。
                在这段代码中，+1e-10的目的是为了防止分母为零的情况。通过添加这个小的常数，即使(data.size(0) - labeled_bs)为零，分母也会变成一个非零的值，避免了除以零的错误。
                这种做法称为数值稳定性处理，可以提高数值计算的稳定性和准确性。在处理概率、损失函数等需要进行除法操作的情况下，这种处理方式是常见的做法。
                '''
                loss = labeled_loss + self.unlabeled_weight()*unlabeled_loss # 将标记样本和无标签样本的损失加权求和得到总损失loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                '''
                如果是测试模式（is_train=False），我们将标记样本的数量labeled_bs设置为当前批次的样本数，
                将标记样本和无标签样本的损失labeled_loss和unlabeled_loss初始化为0，并计算总损失loss
                '''
                labeled_bs = data.size(0)
                labeled_loss = unlabeled_loss = torch.Tensor([0])
                loss = torch.mean(self.loss_fn(outputs, targets))
            labeled_n += labeled_bs

            loop_loss.append(loss.item() / len(data_loader))
            acc = targets.eq(outputs.max(1)[1]).sum().item()
            accuracy.append(acc)
            if print_freq>0 and (batch_idx%print_freq)==0:
                print(f"[{mode}]loss[{batch_idx:<3}]\t labeled loss: {labeled_loss.item():.3f}\t unlabeled loss: {unlabeled_loss.item():.3f}\t loss: {loss.item():.3f}\t Acc: {acc/labeled_bs:.3%}")
            if self.writer:
                self.writer.add_scalar(mode+'_global_loss', loss.item(), self.global_step)
                self.writer.add_scalar(mode+'_global_accuracy', acc/labeled_bs, self.global_step)
        print(f">>>[{mode}]loss\t loss: {sum(loop_loss):.3f}\t Acc: {sum(accuracy)/labeled_n:.3%}")
        if self.writer:
            self.writer.add_scalar(mode+'_epoch_loss', sum(loop_loss), self.epoch)
            self.writer.add_scalar(mode+'_epoch_accuracy', sum(accuracy)/labeled_n, self.epoch)

        return loop_loss, accuracy

    def unlabeled_weight(self):
        '''用于计算无标签样本的权重'''
        alpha = 0.0
        if self.epoch > self.T1:
            alpha = (self.epoch-self.T1) / (self.T2-self.T1)*self.af
            if self.epoch > self.T2:
                alpha = af
        return alpha
        
    def train(self, data_loader, print_freq=20):
        self.model.train() # 将模型设置为训练模式
        with torch.enable_grad(): # 启用梯度计算
            loss, correct = self._iteration(data_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            loss, correct = self._iteration(data_loader, print_freq, is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1):
        for ep in range(epochs):
            self.epoch = ep
            if scheduler is not None:
                scheduler.step()
            print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, print_freq)
            print("------ Testing epochs: {} ------".format(ep))
            self.test(test_data, print_freq)
            if ep % self.save_freq == 0:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                    "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "model_epoch_{}.pth".format(epoch))
