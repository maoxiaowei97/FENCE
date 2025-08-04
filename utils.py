import numpy as np
import torch
from torch.optim import Adam
import nni
import time
from tqdm import tqdm
import logging
import datetime
import csv 
import os
import random 
current_time = datetime.datetime.now()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 1e6
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.path}')

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
    valid_epoch_interval=2,
    savename=""
):
    if savename == "":
        current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        savename = f"model_t_{current_time_str}"
    
    uncond_model_save_path = f"./params/{savename}_uncond.pth"
    cond_model_save_path = f"./params/{savename}_cond.pth"
    
    phase1_epochs = int(config["phase1_epoch"])
    phase2_epochs = int(config["phase2_epoch"])
    
    early_stopper_1 = EarlyStopping(patience=10, verbose=True, path=uncond_model_save_path)
    early_stopper_2 = EarlyStopping(patience=5, verbose=True, path=cond_model_save_path)
    # phase 1
    optimizer_1 = Adam(
        model.diffmodel_uncond.parameters(), 
        lr=float(config["phase1_lr"]), 
        weight_decay=1e-6
    )
    lr_scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_1, milestones=[int(0.75 * phase1_epochs), int(0.9 * phase1_epochs)], gamma=0.1)

    # phase 2
    optimizer_2 = Adam(
        model.diffmodel_cond.parameters(), 
        lr=float(config["phase2_lr"]),
        weight_decay=1e-5
    )
    lr_scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, milestones=[int(0.75 * phase1_epochs), int(0.9 * phase1_epochs)], gamma=0.1)

    train_start = time.time()
    # phase train
    print(f"--- Starting Phase 1: Unconditional Pre-training (Max Epochs: {phase1_epochs}) ---")
    for epoch_no in range(phase1_epochs):
        avg_loss = 0
        model.train() 
        model.diffmodel_uncond.train()
        model.diffmodel_cond.eval()
        
        for batch_no, train_batch in enumerate(train_loader):
            optimizer_1.zero_grad()
            loss = model(train_batch, is_phase1=True) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.diffmodel_uncond.parameters(), max_norm=5.0)
            avg_loss += loss.item()
            optimizer_1.step()
        
        avg_loss /= (batch_no + 1)
        print(f"Phase 1 - Epoch: {epoch_no+1}/{phase1_epochs}, Avg Train Loss: {avg_loss:.6f}, LR: {optimizer_1.param_groups[0]['lr']:.6f}")
        lr_scheduler_1.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                for batch_no_val, valid_batch in enumerate(valid_loader, start=1):
                    loss = model(valid_batch, is_train=0, is_phase1=True)
                    avg_loss_valid += loss.item()
            avg_loss_valid /= batch_no_val
            print(f"Phase 1 - Valid Epoch: {epoch_no}, Avg Valid Loss: {avg_loss_valid:.6f}")
            
            early_stopper_1(avg_loss_valid, model.diffmodel_uncond)
            if early_stopper_1.early_stop:
                print("Early stopping triggered in Phase 1.")
                break
    
    print("--- End of Phase 1 ---")
    print(f"Loading best unconditional model from: {uncond_model_save_path} before starting Phase 2")
    model.diffmodel_uncond.load_state_dict(torch.load(uncond_model_save_path, map_location=model.device))
    model.diffmodel_cond.load_state_dict(torch.load(uncond_model_save_path, map_location=model.device))
    # phase2 train
    print(f"--- Starting Phase 2: Conditional Fine-tuning (Max Epochs: {phase2_epochs}) ---")
    for epoch_no_phase2 in range(phase2_epochs):
        avg_loss = 0
        model.train() 
        model.diffmodel_cond.train()
        model.diffmodel_uncond.eval()
        
        for batch_no, train_batch in enumerate(train_loader):
            optimizer_2.zero_grad()
            loss = model(train_batch, is_phase1=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.diffmodel_cond.parameters(), max_norm=1.0)
            avg_loss += loss.item()
            optimizer_2.step()
        
        avg_loss /= (batch_no + 1)
        print(f"Phase 2 - Epoch: {epoch_no_phase2}/{phase2_epochs-1}, Avg Train Loss: {avg_loss:.6f}, LR: {optimizer_2.param_groups[0]['lr']:.6f}")
        lr_scheduler_2.step()

        if valid_loader is not None and (epoch_no_phase2 + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                for batch_no_val, valid_batch in enumerate(valid_loader, start=1):
                    loss = model(valid_batch, is_train=0, is_phase1=False)
                    avg_loss_valid += loss.item()
            avg_loss_valid /= batch_no_val
            print(f"Phase 2 - Valid Epoch: {epoch_no_phase2}, Avg Valid Loss: {avg_loss_valid:.6f}")
            
            early_stopper_2(avg_loss_valid, model.diffmodel_cond)
            if early_stopper_2.early_stop:
                print("Early stopping triggered in Phase 2.")
                break

    print("Training finished.")
    print(f"Loading best conditional model from: {cond_model_save_path}")
    model.diffmodel_cond.load_state_dict(torch.load(cond_model_save_path, map_location=model.device))
    
    train_end_time = time.time()
    print(f"Total training time: {train_end_time - train_start:.2f} seconds")

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for q in quantiles:
        q_pred = torch.quantile(forecast, q, dim=1) 
        q_loss = quantile_loss(target, q_pred, q, eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model, test_loader, _std,_mean,use_nni, nsample=10, results_file=None):

    test_start = time.time()
    with torch.no_grad():
        model.eval()
        mse_total = 0 
        mae_total = 0 
        mape_total = 0  
        evalpoints_total = 0 
        
        
        all_generated_samples = []
        all_target = []
        all_evalpoint=[]
        all_observed_point=[]
        all_observed_time=[]
        device = next(model.parameters()).device 
        if not isinstance(_std, torch.Tensor):
            scaler = torch.tensor(_std, device=device, dtype=torch.float32)
        else:
            scaler = _std.to(device)

        if not isinstance(_mean, torch.Tensor):
            mean_scaler = torch.tensor(_mean, device=device, dtype=torch.float32)
        else:
            mean_scaler = _mean.to(device)
        print("START TEST...")
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it: 
            for batch_no, test_batch in enumerate(it, start=1):  #
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  
                c_target = c_target.permute(0, 2, 1) 
                eval_points = eval_points.permute(0, 2, 1).long()
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
            miss_rate = model.config['train']['miss_rate']
            guidance_method = model.guidance
            
            if guidance_method == 'cfg':
                guidance_param_value = model.cfg_scale
            elif guidance_method == 'fbg':
                guidance_param_value = model.fbg_mode
            else:
                guidance_param_value = 'N/A'

            file_exists = os.path.isfile(results_file)
            
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow([
                        'miss_rate', 'guidance_method', 'guidance_param_value', 
                        'rmse', 'mae', 'mape','crps'
                    ])
                
                writer.writerow([
                    miss_rate, guidance_method, guidance_param_value,
                    f"{final_rmse:.4f}", f"{final_mae:.4f}", f"{final_mape:.4f}",f"{final_crps:.4f}"
                ])
            print(f"results file saved to: {results_file}")
    test_end_time = time.time()
    print("Testing time:", test_end_time - test_start)
