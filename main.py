import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import transformer.Constants as Constants
# Utils.py 从 SMURF-THP 复用
import Utils as Utils

from preprocess.Dataset import get_dataloader
from transformer.Layers import get_non_pad_mask
from transformer.Models import FlowMatchingTHP  # 引入新模型
from flow_matching.solver import ODESolver  # 引入求解器
from tqdm import tqdm


# (wandb optional)
# import wandb
# wandb.init(project="flow-matching-thp")

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        """ Normal load data. """
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')

    # 计算归一化统计量
    time_flat = []
    for i in train_data:
        # 确保 time_since_last_event > 0
        time_flat += [elem['time_since_last_event'] for elem in i if elem['time_since_last_event'] > 0]

    # 修复：确保在 CPU 上计算统计数据，以防 device 不可用或数据在 CPU
    time_flat_tensor = torch.tensor(time_flat)  # , device=opt.device)

    time_mean = 1.0
    time_std = 1.0

    if opt.normalize == 'normal':
        time_mean = time_flat_tensor.mean().item()
        print(f'[Info] Time Gap Mean (for normalization): {time_mean}')
    elif opt.normalize == 'log':
        log_time = torch.log(time_flat_tensor.clamp(min=1e-8))  # 避免 log(0)
        time_mean = log_time.mean().item()
        time_std = log_time.std().item()
        print(f'[Info] Log Time Gap Mean: {time_mean}, Std: {time_std}')

    opt.time_mean = time_mean
    opt.time_std = time_std

    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    # Load three datasets
    opt.max_len = 0  # Dataloader 会自动设置
    trainloader = get_dataloader(train_data, opt, shuffle=True, split='train')
    devloader = get_dataloader(dev_data, opt, shuffle=False, split='dev')
    testloader = get_dataloader(test_data, opt, shuffle=False, split='test')

    return trainloader, devloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """
    model.train()
    total_loss = 0
    total_num_event = 0
    total_correct = 0  # <-- 新增：用于计算准确率

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        # enc_out: (B, L, D_model), prediction: dict
        enc_out, prediction = model(event_type, event_time, time_gap)

        # compute the loss (FM loss + Type loss)
        loss = model.compute_loss(enc_out, event_time, time_gap, event_type, prediction, None)

        """ update parameters """
        loss.backward()
        optimizer.step()

        # --- (新增) 计算辅助准确率 ---
        with torch.no_grad():  # 不追踪梯度
            type_logits = prediction['type_logits']  # (B, L-1, NumTypes)
            pred_type = type_logits.argmax(dim=-1)  # (B, L-1)
            target_type = event_type[:, 1:]  # (B, L-1), 1-based index

            non_pad_mask = (target_type != Constants.PAD)
            # 比较 0-based pred_type 和 0-based target (target_type - 1)
            correct_preds = (pred_type == (target_type - 1)) & non_pad_mask

            total_correct += correct_preds.sum().item()
        # --- 结束新增 ---

        """ note keeping """
        # 实际参与 loss 计算的事件数
        num_event = (event_type[:, 1:] != Constants.PAD).sum().item()

        # 修复：用 平均loss * 事件数 = 该批次的总loss
        if num_event > 0:
            total_loss += loss.item() * num_event

        total_num_event += num_event

    # (acc 暂时不计算)
    # 修复：避免除以 0
    if total_num_event == 0:
        return 0.0, 0.0

    final_loss = total_loss / total_num_event
    final_acc = total_correct / total_num_event  # <-- 新增：返回计算出的准确率
    return final_loss, final_acc


def eval_epoch(model, validation_data, pred_loss_func, eval_generation, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_loss = 0
    total_num_event = 0
    total_correct = 0  # <-- 新增：用于计算准确率

    # --- 阶段 1: 计算验证集 Loss (同 train_epoch) ---
    if not eval_generation:
        start = time.time()
        with torch.no_grad():
            for batch in tqdm(validation_data, mininterval=2, desc='  - (Validating) ', leave=False):
                event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
                # 修复：参数顺序
                enc_out, prediction = model(event_type, event_time, time_gap)
                loss = model.compute_loss(enc_out, event_time, time_gap, event_type, prediction, None)

                # --- (新增) 计算辅助准确率 ---
                type_logits = prediction['type_logits']  # (B, L-1, NumTypes)
                pred_type = type_logits.argmax(dim=-1)  # (B, L-1)
                target_type = event_type[:, 1:]  # (B, L-1), 1-based index

                non_pad_mask = (target_type != Constants.PAD)
                # 比较 0-based pred_type 和 0-based target (target_type - 1)
                correct_preds = (pred_type == (target_type - 1)) & non_pad_mask

                total_correct += correct_preds.sum().item()
                # --- 结束新增 ---

                num_event = (event_type[:, 1:] != Constants.PAD).sum().item()

                # 修复：用 平均loss * 事件数 = 该批次的总loss
                if num_event > 0:
                    total_loss += loss.item() * num_event

                total_num_event += num_event

        # 修复：避免除以 0
        if total_num_event == 0:
            final_loss = 0.0
            final_acc = 0.0
        else:
            final_loss = total_loss / total_num_event
            final_acc = total_correct / total_num_event  # <-- 新增：计算最终准确率

        print('  - (Validating) loss: {ll: 8.5f}, elapse: {elapse:3.3f} min'
              .format(ll=final_loss, elapse=(time.time() - start) / 60))

        return final_loss, final_acc  # <-- 新增：返回计算出的准确率

    # --- 阶段 2: 逆向生成样本 (SMURF 的 eval_langevin) ---
    if eval_generation:
        print("[Info] Starting reverse generation (ODE Sampling)...")

        # 1. 初始化 ODE Solver
        # 我们需要一个包装器来将 model.v_field 适配给 ODESolver
        # ODESolver 期望 v_field(x, t, c=...)
        # model.v_field 签名是 v_field(x, t, c)
        class VelocityModelWrapper:
            def __init__(self, v_field_func):
                self.v_field = v_field_func

            def __call__(self, x, t, **model_extras):
                # c 是固定的条件
                c = model_extras.get('c')
                # t 是标量
                # x: (B*N, L_gen, D_x)

                # model.v_field 期望 (B, L, D_in), (B, L, 1), (B, L, D_cond)
                return self.v_field(x=x, t=t, c=c)

        solver = ODESolver(VelocityModelWrapper(model.v_field))

        # 修复：确保 opt.eval_quantile 是数组
        if isinstance(opt.eval_quantile, float):
            raise ValueError("opt.eval_quantile is a float, expected array. Check main() logic.")

        total_coverage_single = torch.zeros(len(opt.eval_quantile))
        total_intlen = torch.zeros(len(opt.eval_quantile))
        total_crps = 0
        total_corr_type = 0
        total_num_pred = 0

        start = time.time()

        with torch.no_grad():
            for batch in tqdm(validation_data, mininterval=2, desc='  - (Sampling) ', leave=False):
                """ prepare data """
                event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

                # 1. 获取历史编码 C
                non_pad_mask = get_non_pad_mask(event_type)
                # 修复：参数顺序
                enc_output = model.encoder(event_type, event_time, non_pad_mask)  # (B, L, D_model)

                # 找到每个序列最后一个非 PAD 事件的索引 (用于评估)
                seq_lengths = event_type.ne(Constants.PAD).sum(dim=1)  # (B,)
                # (L-1) 空间中的索引
                last_event_idx = seq_lengths - 2
                last_event_idx[last_event_idx < 0] = 0

                # (B,)
                batch_indices = torch.arange(event_time.shape[0], device=opt.device)

                # 历史 c_{T-1} 用于预测 x_T
                # (B, D_model)
                c_last = enc_output[batch_indices, last_event_idx, :]
                # (B, 1, D_model)
                c_gen = c_last.unsqueeze(1)

                # 准备生成
                B = c_gen.shape[0]
                L_gen = 1  # 一次生成一个

                # 2. 准备先验 x0 (多次采样 N_samples)
                N_samples = opt.n_samples  # e.g., 100

                # (B, L_gen, D_x) -> (B * N_samples, L_gen, D_x)
                x_0 = torch.randn(B * N_samples, L_gen, model.x_dim, device=opt.device)

                # (B, L_gen, D_cond) -> (B * N_samples, L_gen, D_cond)
                c_gen_expanded = c_gen.repeat_interleave(N_samples, dim=0)

                # 3. 定义 ODE 时间
                time_grid = torch.tensor([0.0, 1.0], device=opt.device)

                # 4. 逆向求解
                # x_1_pred: (B * N_samples, L_gen, D_x)
                x_1_pred = solver.sample(
                    x_init=x_0,
                    time_grid=time_grid,
                    method=opt.solver_method,
                    step_size=opt.solver_step_size,
                    c=c_gen_expanded  # 作为 **model_extras 传入
                )

                # (B * N_samples, L_gen, D_x) -> (B, N_samples, L_gen, D_x)
                x_1_pred = x_1_pred.view(B, N_samples, L_gen, model.x_dim)

                # 5. 解码 x_1
                # (B, N_samples, L_gen, 1)
                pred_time_norm = x_1_pred[..., 0:1]
                # (B, N_samples, L_gen, D_latent)
                pred_z_m = x_1_pred[..., 1:]

                # 6. 反归一化时间
                # (B, N_samples, L_gen)
                pred_time_gap = model.denormalize_time(pred_time_norm.squeeze(-1))
                t_sample = pred_time_gap.squeeze(-1)  # (B, N_samples)

                # 7. 解码事件类型 (GMM)
                # (NumTypes, D_latent)
                gmm_means = model.gmm_means.weight

                # (B, N_samples, L_gen, 1, D_latent)
                pred_z_m_exp = pred_z_m.unsqueeze(-2)
                # (1, 1, 1, NumTypes, D_latent)
                gmm_means_exp = gmm_means.view(1, 1, 1, model.num_types, model.latent_dim)

                # (B, N_samples, L_gen, NumTypes)
                distances = ((pred_z_m_exp - gmm_means_exp) ** 2).sum(-1)

                # (B, N_samples, L_gen)
                type_sample_idx = torch.argmin(distances, dim=-1)
                type_sample = type_sample_idx + 1  # 转换回 1-based index

                type_sample = type_sample.squeeze(-1)  # (B, N_samples)

                # 8. 获取 GT (最后一个非 PAD 事件)
                # (B, 1)
                gt_t = time_gap.gather(1, last_event_idx.unsqueeze(1))
                # (B, 1)
                event_type_eval = event_type[:, 1:].gather(1, last_event_idx.unsqueeze(1))

                # 9. 评估 (使用复用的 SMURF Utils)
                # t_sample (B, N_samples), gt_t (B, 1)
                # type_sample (B, N_samples), event_type_eval (B, 1)
                coverage_single, intlen, crps, corr_type = Utils.evaluate_samples(
                    t_sample, gt_t, type_sample, event_type_eval, opt)

                # 累加
                total_coverage_single += coverage_single
                total_intlen += intlen
                total_crps += crps.item()
                total_corr_type += corr_type.item()
                total_num_pred += B  # 评估的序列数

        # --- 报告结果 ---
        # 修复：避免除以 0
        if total_num_pred == 0:
            print("[Warning] No predictions made during evaluation.")
            return {'cs': 0, 'accuracy': 0, 'crps': 0, 'intlen': 0}

        total_coverage_single /= total_num_pred
        total_intlen /= total_num_pred
        total_crps /= total_num_pred
        total_corr_type /= total_num_pred

        big_idx_single = int(len(opt.eval_quantile) / 2)
        cs_single_big = np.sqrt(
            ((total_coverage_single[big_idx_single:] - opt.eval_quantile[big_idx_single:]) ** 2).mean())

        print('  - (Sampling) Accuracy: {type: 8.5f}, Calib-Score: {csb1: 8.5f}, '
              'CRPS: {crps: 8.5f}, Interval Length: {intlen: 8.5f}, elapse: {elapse:3.3f} min'
              .format(type=total_corr_type, csb1=cs_single_big, crps=total_crps,
                      intlen=total_intlen[big_idx_single], elapse=(time.time() - start) / 60))
        print('coverage_single: ', total_coverage_single)
        print('Interval Length: ', total_intlen)

        results = {'cs': cs_single_big.item(), 'accuracy': total_corr_type,
                   'crps': total_crps, 'intlen': total_intlen[big_idx_single]}

        return results  # (返回评估结果)


def train(config=None):
    if True:
        """ prepare dataloader """
        trainloader, devloader, testloader, num_types = prepare_dataloader(config)
        config.num_types = num_types

        """ prepare model """
        model = FlowMatchingTHP(num_types, config)

        # 注入归一化参数
        model.time_mean = config.time_mean
        model.time_std = config.time_std

        model.to(config.device)

        """ optimizer and scheduler """
        if config.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                   config.lr, betas=(0.9, 0.999), eps=1e-5)
        else:
            optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                                  config.lr, momentum=0.9, weight_decay=5e-4)

        if config.scheduler == 'cosLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 64, verbose=True)
        elif config.scheduler == 'reduce':
            # 修复：拼写错误
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

        # pred_loss_func 在模型内部定义
        pred_loss_func = None

        """ number of parameters """
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(num_params))
        # wandb.watch(model)

        """ Start training. """
        best_loss = 1e5  # record the loss to save the best model
        for epoch_i in range(config.epoch):
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            start = time.time()
            train_loss, train_acc = train_epoch(model, trainloader, optimizer, pred_loss_func, config)
            print('  - (Training)    loss: {ll: 8.5f}, accuracy: {type: 8.5f}, elapse: {elapse:3.3f} min'
                  .format(ll=train_loss, type=train_acc, elapse=(time.time() - start) / 60))

            start = time.time()
            valid_loss, valid_acc = eval_epoch(model, devloader, pred_loss_func, False, config)
            # <-- 新增：在打印时使用计算出的 valid_acc
            print('  - (Validating)  loss: {ll: 8.5f}, accuracy: {type: 8.5f}, elapse: {elapse:3.3f} min'
                  .format(ll=valid_loss, type=valid_acc, elapse=(time.time() - start) / 60))

            # wandb.log({'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss})

            ###save the best model
            if valid_loss <= best_loss:
                if config.save_path is not None:
                    print('!!! Saving best model to ' + config.save_path + config.save_name + '.pth!!!')
                    best_loss = valid_loss
                    torch.save({'model': model.state_dict(), 'best_loss': best_loss, 'epoch': epoch_i},
                               config.save_path + config.save_name + '.pth')

            # 评估生成 (SMURF eval)
            if epoch >= config.eval_epoch:
                start = time.time()
                test_results = eval_epoch(model, testloader, pred_loss_func, True, config)
                # wandb.log(test_results)

            if config.scheduler == 'reduce':
                scheduler.step(valid_loss)
            else:
                scheduler.step()


# (eval 函数需要类似地修改以加载模型并调用 eval_epoch(..., eval_generation=True, ...))
def eval(config):
    """ Evaluation phase. """

    # 修复：添加 pickle
    import pickle

    # --- 1. 准备数据 (同 train, 但不需要 trainloader) ---
    _, devloader, testloader, num_types = prepare_dataloader(config)
    config.num_types = num_types

    # --- 2. 准备模型 ---
    model = FlowMatchingTHP(num_types, config)
    model.time_mean = config.time_mean
    model.time_std = config.time_std
    model.to(config.device)

    # --- 3. 加载已保存的 checkpoint ---
    if config.load_path_name:
        print(f"[Info] Loading model from {config.load_path_name}")
        try:
            checkpoint = torch.load(config.load_path_name)
            model.load_state_dict(checkpoint['model'])
            print(
                f"[Info] Model loaded successfully (epoch {checkpoint.get('epoch', 'N/A')}, loss {checkpoint.get('best_loss', 'N/A')}).")
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            return
    else:
        print("[Error] No model specified for evaluation. Use -load_path_name.")
        return

    # pred_loss_func 在模型内部定义
    pred_loss_func = None

    # --- 4. 运行评估 ---
    print("[Info] Starting evaluation...")
    start = time.time()
    # 运行 eval_epoch，并强制开启 eval_generation=True
    test_results = eval_epoch(model, testloader, pred_loss_func, True, config)

    print("[Info] Evaluation finished.")
    # (您可以选择将 test_results 保存到 config.save_result 指定的文件中)
    # 比如：
    # if config.save_result:
    #    print(f"Saving results to {config.save_result}")
    #    with open(config.save_result, 'wb') as f:
    #        pickle.dump(test_results, f)


def main():
    """ Main function. """
    parser = argparse.ArgumentParser()

    #### data option
    parser.add_argument('-data', required=True)
    parser.add_argument('-normalize', type=str, default='log', choices=['None', 'normal', 'log'])
    parser.add_argument('-seed', type=int, default=2023)

    #### training option
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-eval_epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-scheduler', type=str, choices=['cosLR', 'reduce'], default='cosLR')
    parser.add_argument('-loss_lambda', type=float, default=1.0, help="Weight for auxiliary type loss")

    #### model type (固定为新模型)
    # parser.add_argument('-model', type=str, default='flow_thp')

    #### model hyperparameter
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)

    #### (新) Flow Matching & GMM 参数
    parser.add_argument('-d_latent', type=int, default=16, help='Latent dim for GMM')
    parser.add_argument('-fm_sigma', type=float, default=0.1, help='Sigma for Flow Matching path (if used)')
    parser.add_argument('-solver_method', type=str, default='euler', help='ODE solver method')
    parser.add_argument('-solver_step_size', type=float, default=0.01, help='ODE solver step size')

    #### save option and eval option
    parser.add_argument('-save_path', type=str, default=None)
    parser.add_argument('-save_name', type=str, default=None)
    parser.add_argument('-just_eval', action='store_true')
    parser.add_argument('-load_path_name', type=str, default=None)
    parser.add_argument('-save_result', type=str, default=None)

    #### (原 SMURF) sampling option (N_samples 和 eval_quantile 仍在使用)
    parser.add_argument('-n_samples', type=int, default=100)
    parser.add_argument('-eval_quantile', type=float, default=-1)
    parser.add_argument('-eval_quantile_step', type=float, default=0.1)

    opt = parser.parse_args()
    # default device is CUDA
    opt.device = torch.device('cuda')

    # set seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.set_printoptions(precision=4)

    # opt = wandb.config.update(opt) # (wandb optional)
    print('[Info] parameters: {}'.format(opt))

    # 修复：将 eval_quantile 逻辑移到 train/eval 调用之前
    if int(opt.eval_quantile) == -1:
        # 修复：使用 torch.arange 代替 np.arange，并指定 device
        opt.eval_quantile = torch.arange(
            opt.eval_quantile_step, 1, opt.eval_quantile_step,
            device=opt.device, dtype=torch.float32
        )
    else:
        # 修复：使用 'raise' 来真正地停止程序
        raise NotImplementedError("Please set eval_quantile to -1. Other values are not supported.")

    # 修复：删除重复的逻辑，只保留一个调用块
    if not opt.just_eval:
        train(opt)
    else:
        # 现在我们可以调用新添加的 eval 函数了
        eval(opt)
        print("[Info] Evaluation complete.")


if __name__ == '__main__':
    main()

