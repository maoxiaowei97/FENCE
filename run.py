import os
import torch
from args import parse_args
from dataset_traffic import get_dataloader
from main_model import FENCE_Traffic
from utils import train, evaluate, set_seed

def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = args.config_parser

    data_prefix = cfg['file']['data_prefix']
    true_path = os.path.join(data_prefix, f"true_data_{cfg['train']['type']}_{cfg['train']['miss_rate']}_v2.npz")
    miss_path = os.path.join(data_prefix, f"miss_data_{cfg['train']['type']}_{cfg['train']['miss_rate']}_v2.npz")
    loaders = get_dataloader(true_path, miss_path,
                             float(cfg['train']['val_ratio']),
                             float(cfg['train']['test_ratio']),
                             int(cfg['train']['batch_size']),
                             int(cfg['train']['sample_len']))
    train_loader, valid_loader, test_loader, target_dim, _std, _mean = loaders

    model = FENCE_Traffic(cfg, target_dim, args.device).to(args.device)

    if args.mode == 'train':
        train(model, cfg["train"], train_loader, valid_loader, savename=args.savename)
        evaluate(model, test_loader, _std, _mean, args.nni_params,
                 nsample=int(cfg['diffusion']['nsample']), results_file=args.results_file)
    else:
        model.diffmodel_cond.load_state_dict(torch.load(args.cond_path, map_location=args.device))
        model.diffmodel_uncond.load_state_dict(torch.load(args.uncond_path, map_location=args.device))
        model.eval()
        evaluate(model, test_loader, _std, _mean, args.nni_params,
                 nsample=int(cfg['diffusion']['nsample']), results_file=args.results_file)

if __name__ == "__main__":
    main()
