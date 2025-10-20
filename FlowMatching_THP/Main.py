import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader

from tqdm import tqdm
import torch.nn.functional as F
from transformer import FlowMatching as FM  # 假设您的流模型文件名为 FlowMatching.py
from transformer.Models import THPEncoder  # 必须导入 THPEncoder

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(thp_encoder, fm_model, event_mapper, fm_trainer, training_data, optimizer, opt):
    """
    新的 Epoch 操作 (FM 版本)
    """
    thp_encoder.train()
    fm_model.train()
    event_mapper.train()

    total_fm_loss = 0
    total_num_events = 0  # (用于归一化)

    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):

        """ 1. 准备数据 """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        # event_time, time_gap, event_type 形状: (batch_size, seq_len)

        optimizer.zero_grad()

        """ 2. 运行 THP 编码器获取历史上下文 c """
        # c = history encoding (来自 1...n-1)
        c_history = thp_encoder(event_type[:, :-1], event_time[:, :-1])
        # c_history 形状: (batch_size, seq_len-1, d_model)

        """ 3. 准备目标 z """
        # z 是 (time_gap_2, type_2), ..., (time_gap_n, type_n)
        target_time_gap = time_gap[:, 1:]
        target_event_type = event_type[:, 1:]

        # 4. 过滤掉 PAD
        non_pad_mask = target_event_type.ne(Constants.PAD)  # (bs, seq_len-1)

        c_history_flat = c_history[non_pad_mask]  # (num_events, d_model)
        if c_history_flat.shape[0] == 0:
            continue  # 跳过空批次

        target_time_gap_flat = target_time_gap[non_pad_mask]
        target_event_type_flat = target_event_type[non_pad_mask]

        # 5. 将 (time, type) 转换为向量 z
        z_target_flat = event_mapper.encode(
            target_time_gap_flat,
            target_event_type_flat
        )
        # z_target_flat 形状: (num_events, event_dim)

        """ 6. 计算 FM 损失 """
        loss = fm_trainer.get_train_loss(c=c_history_flat, z_target=z_target_flat)

        """ 7. 反向传播和更新 """
        loss.backward()
        optimizer.step()

        """ 8. 记录 """
        total_fm_loss += loss.item() * c_history_flat.shape[0]
        total_num_events += c_history_flat.shape[0]

    if total_num_events == 0:
        return 0.0
    return total_fm_loss / total_num_events


def eval_epoch(thp_encoder, fm_model, event_mapper, fm_trainer, validation_data, opt, num_eval_steps=50):
    """
    新的 Eval 操作 (FM 版本)，包含采样和指标计算
    """
    thp_encoder.eval()
    fm_model.eval()
    event_mapper.eval()

    total_fm_loss = 0
    total_time_se = 0  # 累积时间平方误差
    total_event_correct = 0  # 累积事件类型正确数
    total_num_events = 0  # 总事件数

    # 准备 ODE 模拟器
    ts = torch.linspace(0, 1, num_eval_steps).to(opt.device)

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):

            """ 1-5. 准备 c 和 z (与 train_epoch 相同) """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
            c_history = thp_encoder(event_type[:, :-1], event_time[:, :-1])
            target_time_gap = time_gap[:, 1:]
            target_event_type = event_type[:, 1:]
            non_pad_mask = target_event_type.ne(Constants.PAD)

            c_history_flat = c_history[non_pad_mask]
            if c_history_flat.shape[0] == 0:
                continue

            target_time_gap_flat = target_time_gap[non_pad_mask]
            target_event_type_flat = target_event_type[non_pad_mask]

            z_target_flat = event_mapper.encode(
                target_time_gap_flat,
                target_event_type_flat
            )

            num_events_batch = c_history_flat.shape[0]

            """ 6. 计算验证损失 """
            loss = fm_trainer.get_train_loss(c=c_history_flat, z_target=z_target_flat)
            total_fm_loss += loss.item() * num_events_batch

            """ 7. (评估) 采样生成 z_pred """
            # a. 定义条件 ODE
            ode = FM.LearnedConditionalODE(fm_model, c=c_history_flat)
            simulator = FM.EulerSimulator(ode)

            # b. 从 p_simple 采样 x_0
            x_0 = fm_trainer.path.p_simple.sample(num_events_batch).to(opt.device)

            # c. 求解 ODE
            z_pred = simulator.simulate(x_0, ts)  # (num_events, event_dim)

            # d. 解码 z_pred
            # time_gap_pred: (num_events,)
            # type_logits: (num_events, num_types)
            time_gap_pred, type_logits_pred = event_mapper.decode(z_pred)

            """ 8. (评估) 计算 RMSE 和 ACC """
            # a. Time RMSE
            se = (time_gap_pred - target_time_gap_flat) ** 2
            total_time_se += se.sum().item()

            # b. Type Accuracy
            # (num_events,)
            type_pred = type_logits_pred.argmax(dim=-1) + 1  # 转换回 1-based
            correct = (type_pred == target_event_type_flat).float()
            total_event_correct += correct.sum().item()

            total_num_events += num_events_batch

    if total_num_events == 0:
        return 0.0, 0.0, 0.0

    avg_loss = total_fm_loss / total_num_events
    rmse = np.sqrt(total_time_se / total_num_events)
    accuracy = total_event_correct / total_num_events

    return avg_loss, rmse, accuracy


def train(thp_encoder, fm_model, event_mapper, fm_trainer,
          training_data, validation_data, optimizer, scheduler, opt):
    """ Start training. """

    valid_losses = []  # validation FM loss
    valid_rmses = []  # validation event time prediction RMSE
    valid_accs = []  # validation event type prediction accuracy

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_loss = train_epoch(
            thp_encoder, fm_model, event_mapper, fm_trainer, training_data, optimizer, opt)
        print('  - (Training)    Loss: {loss: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(loss=train_loss, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_rmse, valid_acc = eval_epoch(
            thp_encoder, fm_model, event_mapper, fm_trainer, validation_data, opt)
        print('  - (Validation)  Loss: {loss: 8.5f}, RMSE: {rmse: 8.5f}, '
              'Accuracy: {acc: 8.5f}, elapse: {elapse:3.3f} min'
              .format(loss=valid_loss, rmse=valid_rmse, acc=valid_acc, elapse=(time.time() - start) / 60))

        valid_losses += [valid_loss]
        valid_rmses += [valid_rmse]
        valid_accs += [valid_acc]

        print('  - [Info] Minimum validation loss: {loss: 8.5f}, '
              'Minimum RMSE: {rmse: 8.5f}, Maximum accuracy: {acc: 8.5f}'
              .format(loss=min(valid_losses), rmse=min(valid_rmses), acc=max(valid_accs)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {loss: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, loss=valid_loss, acc=valid_acc, rmse=valid_rmse))

        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    # (保留所有原始参数)
    parser.add_argument('-data', required=True)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-log', type=str, default='log.txt')

    # (移除 -smooth 参数，FM 不需要它)
    # parser.add_argument('-smooth', type=float, default=0.1)

    # (新) FM 相关参数
    parser.add_argument('-event_dim', type=int, default=64, help='Dimension of the continuous event vector z')
    parser.add_argument('-time_emb_dim_mapper', type=int, default=16, help='Dimension for time_gap in z')
    parser.add_argument('-time_emb_dim_fm', type=int, default=64, help='Dimension for t in FM model')
    parser.add_argument('-hidden_dim_fm', type=int, default=128, help='Hidden dim of FM MLP')

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Val_Loss, Accuracy, RMSE\n')  # 更新日志表头

    print('[Info] parameters: {}'.format(opt))

    """ 1. prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)
    opt.num_types = num_types  # 保存 num_types

    """ 2. prepare models """

    # a. THP 历史编码器
    thp_encoder = THPEncoder(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    thp_encoder.to(opt.device)

    # b. 事件 <-> 向量 z 转换器
    event_mapper = FM.EventTargetMapper(
        event_dim=opt.event_dim,
        num_types=num_types,
        time_emb_dim=opt.time_emb_dim_mapper
    )
    event_mapper.to(opt.device)

    # c. 条件流模型 v(x, t, c)
    fm_model = FM.ConditionalMLPVectorField(
        event_dim=opt.event_dim,
        d_model=opt.d_model,  # (来自 thp_encoder)
        time_emb_dim=opt.time_emb_dim_fm,
        hidden_dim=opt.hidden_dim_fm
    )
    fm_model.to(opt.device)

    """ 3. prepare FM path and trainer """

    path = FM.GaussianConditionalProbabilityPath(
        event_dim=opt.event_dim,
        alpha=FM.LinearAlpha(),
        beta=FM.SquareRootBeta()
    )

    fm_trainer = FM.ConditionalFlowMatchingTrainer(fm_model, path)
    fm_trainer.to(opt.device)

    """ 4. optimizer and scheduler """
    # (重要) 优化器需要包含所有三个模型的参数
    all_params = (
            list(thp_encoder.parameters()) +
            list(event_mapper.parameters()) +
            list(fm_model.parameters())
    )

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, all_params),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ 5. (移除) 旧的 loss function """
    # (pred_loss_func 已被 fm_trainer 替代)

    """ 6. number of parameters """
    num_params = sum(p.numel() for p in all_params if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ 7. train the model """
    train(thp_encoder, fm_model, event_mapper, fm_trainer,
          trainloader, testloader, optimizer, scheduler, opt)


if __name__ == '__main__':
    main()
