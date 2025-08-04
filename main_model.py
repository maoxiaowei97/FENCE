import numpy as np
import torch
import torch.nn as nn
from diff_models import diff


class FENCE_base(nn.Module):
    def _calculate_fbg_offset(self, t0, verbose=True):
        lambda_ref = self.max_guidance
        
        if verbose:
            print(f'(FBG Info) t0={t0} used to calculate offset. t0 is at timestep {round((self.num_steps - 1) * t0, 2)} of {self.num_steps - 1}')

        log_term = np.log((1 - self.pi) * lambda_ref / (lambda_ref - 1))
        offset = (1 / ((1 - t0) * self.num_steps)) * log_term
        
        if abs(offset) < 1e-9:
            print("Warning: Calculated offset is extremely close to zero.")
            
        return round(offset, 5)

    def _calculate_fbg_temp(self, offset, t1, alpha=10., verbose=True):
        if verbose:
            print(f'(FBG Info) t1={t1} used to calculate temp. t1 is at timestep {round((self.num_steps - 1) * t1, 2)} of {self.num_steps - 1}')

        sigma_sq_history = ((1.0 - self.alpha[:-1]) / (1.0 - self.alpha[1:])) * self.beta[1:]
        sigma_sq_history = np.insert(sigma_sq_history, 0, 0.0)
        
        sigma_sq_history_rev = np.flip(sigma_sq_history)

        t1_idx_float = t1 * (self.num_steps - 1)
        t1_lower_idx = int(np.floor(t1_idx_float))
        
        if t1_lower_idx >= self.num_steps - 1:
            sigma_sq_at_t1 = sigma_sq_history_rev[-1]
        else:
            a = t1_idx_float - t1_lower_idx
            val_lower = sigma_sq_history_rev[t1_lower_idx]
            val_upper = sigma_sq_history_rev[t1_lower_idx + 1]
            sigma_sq_at_t1 = (1 - a) * val_lower + a * val_upper
        
        temp = abs(2 * sigma_sq_at_t1 / alpha * offset)
        return round(temp, 5)

    def __init__(self, target_dim, config,device):
        super().__init__()
        self.target_dim = target_dim
        self.device =device
        self.config = config
        self.emb_time_dim = int(config["model"]["timeemb"])
        self.emb_feature_dim = int(config["model"]["featureemb"])
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1 
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = str(self.emb_total_dim)

        input_dim = 2 
        self.diffmodel_cond = diff(config_diff, input_dim)
        self.diffmodel_uncond = diff(config_diff, input_dim)

        self.num_steps = int(config_diff["num_steps"])
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                float(config_diff["beta_start"]) ** 0.5, float(config_diff["beta_end"]) ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                float(config_diff["beta_start"]), float(config_diff["beta_end"]), self.num_steps
            )

        self.alpha_hat = 1 - self.beta 
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(device).unsqueeze(1).unsqueeze(1) 

        # cfg related params
        self.phase = 'cond'
        self.p_drop = float(config["model"].get("p_drop", 0.1))
        self.cfg_scale = float(config["model"].get("cfg_scale", 5.0))
        # fbg related params
        self.guidance = config["model"]["guidance"]
        if self.guidance == 'fbg':
            fbg_config = config["fbg"]
            self.pi = float(fbg_config["pi"])
            self.fbg_mode = fbg_config["mode"]
            self.constant_guidance = float(fbg_config["constant_guidance"])
            self.max_guidance = float(fbg_config["max_guidance"])
            self.minimal_log_posterior = np.log(( (1-self.pi) * (self.max_guidance - self.constant_guidance + 1)) / (self.max_guidance - self.constant_guidance)) 
            print('Minimum log posterior value: ', self.minimal_log_posterior)

            if "temp" in fbg_config and "offset" in fbg_config:
                self.temp = float(fbg_config["temp"])
                self.offset = float(fbg_config["offset"])
                print(f"FBG: Loaded temp={self.temp} and offset={self.offset} directly from config.")
            elif "t0" in fbg_config and "t1" in fbg_config:
                t0 = float(fbg_config["t0"])
                t1 = float(fbg_config["t1"])
                self.offset = self._calculate_fbg_offset(t0)
                self.temp = self._calculate_fbg_temp(self.offset, t1)
                print(f"FBG: Calculated from t0={t0}, t1={t1} -> temp={self.temp}, offset={self.offset}")
            else:
                raise ValueError("FBG config error: Must provide either ('temp', 'offset') or ('t0', 't1').")

            if self.fbg_mode == 'cluster':
                self.n_clusters = int(fbg_config["n_clusters"])
                print(f"FBG mode: 'cluster' with n_clusters={self.n_clusters}")
        
    def time_embedding(self, pos, d_model=128): 
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device) 
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe 

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand() 
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):  
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(   
            torch.arange(self.target_dim).to(self.device)
        ) 
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  
        side_info = side_info.permute(0, 3, 2, 1)  

        side_mask = cond_mask.unsqueeze(1) 
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid( 
        self, observed_data, cond_mask, observed_mask, side_info, is_train, is_phase1
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, is_phase1, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, is_phase1,set_t=-1):
            B, K, L = observed_data.shape
            if not is_train:  # for validation
                t = (torch.ones(B) * set_t).long().to(self.device)
            else:
                t = torch.randint(0, self.num_steps, [B]).to(self.device)

            current_alpha = self.alpha_torch[t]
            noise = torch.randn_like(observed_data)
            noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

            if is_phase1: # phase 1: unconditional
                input_uncond = torch.cat([torch.zeros_like(noisy_data).unsqueeze(1), noisy_data.unsqueeze(1)], dim=1)
                null_side_info = side_info.clone()
                null_side_info[:, -1:, :, :] = 0
                
                predicted, _ = self.diffmodel_uncond(input_uncond, null_side_info, t)
                
                target_mask = observed_mask
            else: # phase 2: conditional
                input_cond = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
                
                predicted, _ = self.diffmodel_cond(input_cond, side_info, t)

                target_mask = observed_mask - cond_mask

            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

            return loss


    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  
        return total_input

    def update_logp(self, prev_log_posterior, diff, sigma_sq, cluster_labels=None):
        B, K = diff.shape
        if self.fbg_mode in["global","spatial"]:   
            log_posterior = prev_log_posterior - self.temp / (2 * sigma_sq) * diff + self.offset 
        elif self.fbg_mode == "cluster":
            cluster_diff = torch.zeros(B, self.n_clusters, device=self.device)   
            cluster_diff.scatter_add_(dim=1, index=cluster_labels, src=diff)
            log_posterior = prev_log_posterior - self.temp / (2 * sigma_sq) * cluster_diff + self.offset
        return torch.clamp(log_posterior, min=self.minimal_log_posterior, max=3.0)

    def _kmeans_torch_cluster(self, features, n_clusters, num_iter=20):
        B, K, D = features.shape
        device = features.device

        centroids = torch.zeros(B, n_clusters, D, device=device)
        initial_indices = torch.randint(0, K, (B,), device=device)
        centroids[:, 0, :] = features[torch.arange(B), initial_indices, :]
        dists = torch.cdist(features, centroids[:, 0, :].unsqueeze(1)).squeeze(2) # (B, K)

        for i in range(1, n_clusters):
            next_centroid_indices = torch.multinomial(dists, 1).squeeze(1) # (B,)
            centroids[:, i, :] = features[torch.arange(B), next_centroid_indices, :]
            new_dists = torch.cdist(features, centroids[:, i, :].unsqueeze(1)).squeeze(2)
            dists = torch.minimum(dists, new_dists)

        for i in range(num_iter):
            dists = torch.cdist(features, centroids)  # (B, K, n_clusters)
            cluster_assignments = torch.argmin(dists, dim=2) # (B, K)
            assignment_one_hot = nn.functional.one_hot(cluster_assignments, n_clusters).float()
            cluster_counts = assignment_one_hot.sum(dim=1)
            cluster_counts.clamp_(min=1.0)
            new_centroids = torch.bmm(assignment_one_hot.transpose(1, 2), features)
            new_centroids /= cluster_counts.unsqueeze(2)
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                print(f"KMeans converged at iteration {i}")
                break
            centroids = new_centroids
        return cluster_assignments

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        null_side_info = side_info.clone()
        null_side_info[:, -1:, :, :] = 0
        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            # generate noisy observation for unconditional model
            noisy_obs = observed_data.clone()
            noisy_cond_history = []
            for t in range(self.num_steps):
                noise = torch.randn_like(noisy_obs)
                noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                noisy_cond_history.append(noisy_obs * cond_mask)

            # FBG : init log_posterior
            cluster_labels = None 
            if self.guidance == "fbg":
                if self.fbg_mode == 'global':
                    log_posterior = torch.zeros(B, device=self.device) 
                elif self.fbg_mode == 'spatial':
                    log_posterior = torch.zeros(B, K, device = self.device)
                elif self.fbg_mode == 'cluster':
                    log_posterior = torch.zeros(B, self.n_clusters, device=self.device)

            for t in range(self.num_steps - 1, -1, -1):
                # 使用 self.diffmodel_uncond 
                noisy_observed_uncond = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                diff_input_uncond = torch.cat([torch.zeros_like(noisy_observed_uncond).unsqueeze(1),
                                                noisy_observed_uncond.unsqueeze(1)], dim=1)
                predicted_uncond, _ = self.diffmodel_uncond(diff_input_uncond, null_side_info, torch.tensor([t]).to(self.device))
                
                # 使用 self.diffmodel_cond 
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input_cond = torch.cat([cond_obs, noisy_target], dim=1)
                predicted_cond, attn_cond = self.diffmodel_cond(diff_input_cond, side_info, torch.tensor([t]).to(self.device))
                                    
                # self.cluster_update_interval = 1 
                if self.guidance == 'fbg' and self.fbg_mode == 'cluster' and attn_cond is not None:
                    with torch.no_grad():
                        cluster_labels = self._kmeans_torch_cluster(attn_cond, self.n_clusters, num_iter=10)

                # cfg / fbg
                if self.guidance == "cfg":
                    predicted = predicted_uncond + self.cfg_scale * (predicted_cond - predicted_uncond)
                elif self.guidance == "fbg":
                    guidance_scale = torch.exp(log_posterior) / (torch.exp(log_posterior) - (1 - self.pi))
                    guidance_scale += self.constant_guidance - 1
                    if self.fbg_mode == 'global':
                        guidance_scale = guidance_scale.view(B, 1, 1)
                    elif self.fbg_mode == 'spatial':
                        guidance_scale = guidance_scale.view(B, K, 1)
                    elif self.fbg_mode == 'cluster':
                        guidance_scale_spatial = torch.gather(guidance_scale, 1, cluster_labels)
                        guidance_scale = guidance_scale_spatial.view(B, K, 1)
                        
                    guidance_scale = torch.clamp(guidance_scale, min=1.0, max=self.max_guidance)
                    predicted = predicted_uncond + guidance_scale * (predicted_cond - predicted_uncond)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                next_sample = coeff1 * (current_sample - coeff2 * predicted)

                sigma_sq = 0.0
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    next_sample += sigma * noise
                    sigma_sq = sigma ** 2

                # update log_posterior
                if self.guidance == 'fbg' and sigma_sq > 1e-8:
                    with torch.no_grad():
                        uncond_predicted_mean = coeff1 * (current_sample - coeff2 * predicted_uncond)
                        cond_predicted_mean = coeff1 * (current_sample - coeff2 * predicted_cond)

                        if self.fbg_mode == 'global':
                            uncond_mse = torch.sum((next_sample - uncond_predicted_mean)**2, dim=[1, 2])
                            cond_mse = torch.sum((next_sample - cond_predicted_mean)**2, dim=[1, 2])
                        else:
                            uncond_mse = torch.sum((next_sample - uncond_predicted_mean)**2, dim=2)
                            cond_mse = torch.sum((next_sample - cond_predicted_mean)**2, dim=2)

                        diff = cond_mse - uncond_mse
                    log_posterior = self.update_logp(log_posterior, diff, sigma_sq, cluster_labels)
                current_sample = next_sample
            imputed_samples[:, i] = current_sample.detach() 
                
        return imputed_samples

    def forward(self, batch, is_train=1, is_phase1=True):
        (observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, _,) = self.process_data(batch, is_train)  
        
        if self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, is_phase1)

    def evaluate(self, batch, n_samples):
        (observed_data, observed_mask, observed_tp, gt_mask, _, cut_length,) = self.process_data(batch,0)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            for i in range(len(cut_length)):
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

class FENCE_Traffic(FENCE_base):
    def __init__(self, config,  target_dim,device):
        super(FENCE_Traffic, self).__init__(target_dim, config,device)
        self.device=device

    def process_data(self, batch,is_train=1):
        observed_data = batch["observed_data"].to(self.device).float() 
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1) 
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )