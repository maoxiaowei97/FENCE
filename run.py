import os
import torch

from args import parse_args
from dataset_traffic import get_dataloader
from main_model import FENCE_Traffic
from utils import train, evaluate, set_seed
import logging
import sys

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.logfile:
        log_dir = os.path.dirname(args.logfile)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(args.logfile, mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f"Logs will be output to both console and file: {args.logfile}")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logging.info("Logs will only be output to the console.")

    cfg = args.config_parser

    miss_type = cfg['train']['type']
    miss_rate = cfg['train']['miss_rate']
    dataset_name = args.dataset_name
    savename_to_use = args.savename
    results_file_to_use = args.results_file

    logging.info("\n" + "=" * 50)
    logging.info("CSDI Model Run Configuration")
    logging.info("=" * 50)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Dataset: {dataset_name} (loaded from '{args.config}')")
    logging.info(f"Missing type: {miss_type}")
    logging.info(f"Missing rate: {miss_rate}")
    guidance_val = cfg['model']['guidance'] if cfg['model']['guidance'] else 'None'
    logging.info(f"Guidance method: {guidance_val}")
    logging.info(f"epoch phase1: {cfg['train']['phase1_epoch']}")
    logging.info(f"epoch phase2: {cfg['train']['phase2_epoch']}")
    logging.info(f"Model save name: {savename_to_use}")
    logging.info(f"Results file: {results_file_to_use}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Random seed: {args.seed}")
    logging.info(f"phase1 device: {cfg['model'].get('uncond_device', 'N/A')}")
    logging.info(f"phase2 device: {cfg['model'].get('cond_device', 'N/A')}")
    logging.info(f"phase1_lr: {cfg['train']['phase1_lr']}")
    logging.info(f"phase2_lr: {cfg['train']['phase2_lr']}")

    if 'fbg' in cfg:
        logging.info(f"max guidance: {cfg['fbg'].get('max_guidance', 'N/A')}")
        logging.info(f" pi: {cfg['fbg'].get('pi', 'N/A')}")
        logging.info(f" t0: {cfg['fbg'].get('t0', 'N/A')}")
        logging.info(f" t1: {cfg['fbg'].get('t1', 'N/A')}")
        logging.info(f" n_clusters: {cfg['fbg'].get('n_clusters', 'N/A')}")

    logging.info("=" * 50 + "\n")

    data_prefix = cfg['file']['data_prefix']
    true_path = os.path.join(data_prefix, f"true_data_{cfg['train']['type']}_{cfg['train']['miss_rate']}.npz")
    miss_path = os.path.join(data_prefix, f"miss_data_{cfg['train']['type']}_{cfg['train']['miss_rate']}.npz")
    logging.info(f"Preparing data...")
    logging.info(f"  -> Reading true data: {true_path}")
    logging.info(f"  -> Reading missing data: {miss_path}")
    loaders = get_dataloader(true_path, miss_path,
                             float(cfg['train']['val_ratio']),
                             float(cfg['train']['test_ratio']),
                             int(cfg['train']['batch_size']),
                             int(cfg['train']['sample_len']))
    train_loader, valid_loader, test_loader, target_dim, _std, _mean = loaders
    logging.info("Data loading complete.")
    model = FENCE_Traffic(cfg, target_dim, args.device).to(args.device)

    if args.mode == 'train':
        logging.info(f"Starting training mode...")
        train(model, cfg["train"], train_loader, valid_loader, savename=args.savename)
        logging.info("Training finished, starting direct evaluation...")
        evaluate(model, test_loader, _std, _mean, args.nni_params,
                 nsample=int(cfg['diffusion']['nsample']), results_file=args.results_file)
    else:
        logging.info(f"Starting evaluation (evaluation mode only)...")
        logging.info(f"Loading conditional model from path: {args.cond_path}")
        model.diffmodel_cond.load_state_dict(torch.load(args.cond_path, map_location=args.device))
        logging.info("Conditional model loaded successfully.")
        logging.info(f"Loading unconditional model from path: {args.uncond_path}")
        model.diffmodel_uncond.load_state_dict(torch.load(args.uncond_path, map_location=args.device))
        logging.info("Unconditional model loaded successfully.")
        model.eval()
        evaluate(model, test_loader, _std, _mean, args.nni_params,
                 nsample=int(cfg['diffusion']['nsample']), results_file=args.results_file)

if __name__ == "__main__":
    main()