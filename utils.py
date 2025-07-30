import numpy as np
import torch
from torch.optim import Adam
import nni
import time
from tqdm import tqdm
import logging
import datetime
import csv # 导入csv模块
import os
# Get the current time
current_time = datetime.datetime.now()

class EarlyStopping:
    """在验证损失不再改善时提前停止训练。"""
    def __init__(self, patience=4, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): 在停止训练前，等待多少个epoch验证损失没有改善。
                            (连续5个epoch效果不好，所以patience=5)
            verbose (bool): 如果为True，则为每次验证损失的改善打印一条信息。
            delta (float): 被认为是改善的最小变化量。
            path (str): 保存最佳模型的路径。
            trace_func (function): 用于打印信息的函数。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 1e8
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        将该对象作为函数调用。
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # 损失没有改善或改善程度小于delta
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 损失得到改善
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """当验证损失减少时，保存模型。"""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # 确保目录存在
        save_dir = os.path.dirname(self.path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    savename=""
):
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), weight_decay=1e-6)
    p1 = int(0.75 * int(config["epochs"]))
    p2 = int(0.9 * int(config["epochs"]))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    # --- 1. 定义两阶段模型保存路径 ---
    if savename == "":
        current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        savename = f"model_t_{current_time_str}"
    
    # 无条件模型权重路径 (只保存diffmodel子模块)
    uncond_model_save_path = f"./params/{savename}_uncond.pth"
    # 有条件模型路径 (保存整个CSDI模型)
    cond_model_save_path = f"./params/{savename}_cond.pth"
    
    # 定义总epoch数和阶段分割点
    total_epochs = int(config["epochs"])
    phase1_end_epoch = int(config["phase1_epoch"])
    
    # EarlyStopping 只在第二阶段（有条件微调）使用
    early_stopper = EarlyStopping(patience=5, verbose=True, path=cond_model_save_path)

    train_start = time.time()
    print("Training started...")
    for epoch_no in range(total_epochs):
        # 打印当前训练阶段
        if epoch_no == 0:
            print(f"--- Starting Phase 1: Unconditional Pre-training (Epochs 0 to {phase1_end_epoch - 1}) ---")
        elif epoch_no == phase1_end_epoch:
            print(f"--- Starting Phase 2: Conditional Fine-tuning (Epochs {phase1_end_epoch} to {total_epochs - 1}) ---")

        # --- 2. 训练循环 ---
        avg_loss = 0
        model.train()
        for batch_no, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            # model.forward 会根据 epoch_no 自动选择损失函数
            loss = model(train_batch, epoch_no)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()

        avg_loss /= (batch_no + 1)
        print(f"Train Epoch: {epoch_no}, Avg Loss: {avg_loss:.6f}")
        lr_scheduler.step()

        # --- 3. 在第一阶段结束后，保存无条件模型权重 ---
        if (epoch_no + 1) == phase1_end_epoch:
            print(f"--- End of Phase 1 ---")
            print(f"Saving unconditional model weights to {uncond_model_save_path}")
            save_dir = os.path.dirname(uncond_model_save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 只保存 diffmodel_cond 的权重，它在此阶段是纯粹的无条件模型
            torch.save(model.diffmodel_cond.state_dict(), uncond_model_save_path)
            # 重置 early_stopper，为第二阶段的验证做准备
            print("Resetting EarlyStopping for Phase 2.")
            early_stopper = EarlyStopping(patience=5, verbose=True, path=cond_model_save_path)


        # --- 4. 在第二阶段进行验证和早停 ---
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0 and epoch_no >= phase1_end_epoch:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                for batch_no_val, valid_batch in enumerate(valid_loader, start=1):
                    loss = model(valid_batch, epoch_no, is_train=0)
                    avg_loss_valid += loss.item()
            avg_loss_valid /= batch_no_val
            print(f"Valid Epoch: {epoch_no}, Avg Loss: {avg_loss_valid:.6f}")

            # 调用 EarlyStopping，它会检查是否需要停止并自动保存最佳的有条件模型
            early_stopper(avg_loss_valid, model)
            if early_stopper.early_stop:
                print("Early stopping triggered in Phase 2.")
                break

    # --- 5. 训练结束后，加载最佳权重以供后续评估 ---
    print("Training finished.")
    print(f"Loading best conditional model from: {cond_model_save_path}")
    model.load_state_dict(torch.load(cond_model_save_path))
    
    # 同样加载已保存的无条件模型权重到 `diffmodel_uncond`
    # 确保即使在 "train then eval" 模式下，模型也处于正确的推理状态
    if os.path.exists(uncond_model_save_path):
        print(f"Loading unconditional model weights from: {uncond_model_save_path}")
        uncond_weights = torch.load(uncond_model_save_path)
        model.diffmodel_uncond.load_state_dict(uncond_weights)
    else:
        print(f"Warning: Unconditional model file not found at {uncond_model_save_path}. Inference might fail.")

    train_end_time = time.time()
    print(f"Total training time: {train_end_time - train_start:.2f} seconds")

def quantile_loss(target, forecast, quantile, eval_points):
    # 计算预测误差
    error = target - forecast
    # L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    loss = torch.max((quantile - 1) * error, quantile * error)
    # 只计算需要插补的点
    loss = loss * eval_points
    return loss.sum()

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target_unscaled = target * scaler + mean_scaler
    forecast_unscaled = forecast * scaler + mean_scaler

    # 步骤 2: 计算分母（在原始尺度上计算）
    denom = torch.sum(torch.abs(target * eval_points))
    
    # 如果没有评估点，CRPS为0
    if denom == 0:
        return 0.0

    # 步骤 3: 定义要计算的分位数
    quantiles = np.arange(0.05, 1.0, 0.05)
    CRPS = 0
    
    # 步骤 4: 循环计算每个分位数的损失
    for q in quantiles:
        q_pred = torch.quantile(forecast_unscaled, q, dim=1)
        
        q_loss = quantile_loss(target_unscaled, q_pred, q, eval_points)
        CRPS += q_loss

    # 步骤 5: 返回平均 CRPS
    return (CRPS / denom).item() / len(quantiles)

def evaluate(model, test_loader, _std,_mean,use_nni, nsample=10, results_file=None):

    test_start = time.time()
    with torch.no_grad():
        model.eval()
        mse_total = 0 #初始化总均方误差
        mae_total = 0 #初始化平均绝对误差
        mape_total = 0  #初始化总平均绝对百分百误差
        evalpoints_total = 0 #初始化总评估点数
        
        
        all_generated_samples = []
        all_target = []
        all_evalpoint=[]
        all_observed_point=[]
        all_observed_time=[]
        print("START TEST...")
        scaler=_std
        mean_scaler=_mean
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it: #根据batch-size计算
            for batch_no, test_batch in enumerate(it, start=1):  #
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1).long()#(B,L,K)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                
                mse_current = ( ((samples_median.values - c_target) * eval_points) ** 2) * (scaler ** 2)
                mae_current = (torch.abs((samples_median.values - c_target) * eval_points)  ) * scaler
                mape_current = torch.divide(torch.abs((samples_median.values - c_target)*scaler)
                                            ,(c_target*scaler+mean_scaler)*((c_target*scaler+mean_scaler)>(1e-4)))\
                                    .nan_to_num(posinf=0,neginf=0,nan=0)*eval_points
                                     
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mape_total += mape_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "mape_total": mape_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
                logging.info("rmse_total={}".format(np.sqrt(mse_total / evalpoints_total)))
                logging.info("mae_total={}".format(mae_total / evalpoints_total))
                logging.info("mape_total={}".format(mape_total / evalpoints_total))
                logging.info("batch_no={}".format(batch_no))
                    
        final_rmse = np.sqrt(mse_total / evalpoints_total)
        final_mae = mae_total / evalpoints_total
        final_mape = mape_total / evalpoints_total
        final_target = torch.cat(all_target, dim=0)
        final_samples = torch.cat(all_generated_samples, dim=0)
        final_evalpoint = torch.cat(all_evalpoint, dim=0)
        final_crps = calc_quantile_CRPS(
            final_target, final_samples, final_evalpoint, mean_scaler, scaler)
        print(f"RMSE: {final_rmse}")
        print(f"MAE: {final_mae}")
        print(f"MAPE: {final_mape}")
        print(f"CRPS: {final_crps:.4f}")
            
        if results_file:
            # 从模型配置中获取当前评估的参数
            miss_rate = model.config['train']['miss_rate']
            guidance_method = model.guidance
            
            # 根据指导方法确定参数值
            if guidance_method == 'cfg':
                guidance_param_value = model.cfg_scale
            elif guidance_method == 'fbg':
                guidance_param_value = model.fbg_mode
            else:
                guidance_param_value = 'N/A'

            # 检查文件是否存在，如果不存在则写入表头
            file_exists = os.path.isfile(results_file)
            
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow([
                        'miss_rate', 'guidance_method', 'guidance_param_value', 
                        'rmse', 'mae', 'mape','crps'
                    ])
                
                # 写入当前评估的结果
                writer.writerow([
                    miss_rate, guidance_method, guidance_param_value,
                    f"{final_rmse:.4f}", f"{final_mae:.4f}", f"{final_mape:.4f}",f"{final_crps:.4f}"
                ])
            print(f"结果已记录到: {results_file}")
    test_end_time = time.time()
    print("Testing time:", test_end_time - test_start)
