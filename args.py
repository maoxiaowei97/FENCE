import argparse
import os
import configparser
import nni

def parse_args():
    parser = argparse.ArgumentParser(description="FENCE Time Series Imputation")

    parser.add_argument(
        "--config",
        type=str,
        default="config/PEMS08.conf",
        help="Path to the base config file. Dataset name is inferred from this path."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Shortcut for specifying dataset (e.g., 'PEMS08'). Overrides --config."
    )
    parser.add_argument("--miss_type", type=str, default=None,
                        help="Missing pattern (rm, bm), overrides config file.")
    parser.add_argument('--miss_rate', type=str, default=None,
                        help="Missing rate, overrides config file.")
    parser.add_argument('--device', default='cuda:0',
                        help='Device to use, e.g. "cuda:0" or "cpu".')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help='Run mode: "train" or "eval".')
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--guidance', type=str, default=None,
                        choices=['cfg', 'fbg'], help='Guidance method (cfg/fbg).')
    parser.add_argument('--cfg_scale', type=float, default=None,
                        help='(eval only) CFG scale.')
    parser.add_argument('--fbg_mode', type=str, default=None,
                        choices=['global','spatial','cluster'],
                        help='(eval only) FBG mode.')
    parser.add_argument("--savename", type=str, default="",
                        help="Custom save name for the model.")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Custom path for results CSV.")
    parser.add_argument("--cond_path", type=str, default="",
                        help="(eval only) Conditional model weights path.")
    parser.add_argument('--uncond_path', type=str, default="",
                        help="(eval only) Unconditional model weights path.")
    parser.add_argument("--targetstrategy", type=str, default="mix",
                        choices=["mix","random","historical"])
    
    args = parser.parse_args()

    if args.dataset:
        args.config = f"config/{args.dataset}.conf"
        args.dataset_name = args.dataset
    else:
        args.dataset_name = os.path.splitext(os.path.basename(args.config))[0]

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    if args.miss_type:   cfg["train"]["type"]      = args.miss_type
    if args.miss_rate:   cfg["train"]["miss_rate"] = args.miss_rate
    if args.guidance:    cfg["model"]["guidance"]   = args.guidance
    if args.cfg_scale:   cfg["model"]["cfg_scale"]  = str(args.cfg_scale)
    if args.fbg_mode:
        if "fbg" not in cfg:  cfg.add_section("fbg")
        cfg["fbg"]["mode"] = args.fbg_mode
    cfg["model"]["target_strategy"] = args.targetstrategy

    miss_type = cfg['train']['type']
    miss_rate = cfg['train']['miss_rate']
    base = f"{args.dataset_name}_{miss_type}_{miss_rate}"
    args.savename = args.savename or base
    if not args.results_file:
        os.makedirs("results", exist_ok=True)
        args.results_file = os.path.join("results", f"{base}.csv")

    if int(cfg['train'].get('use_nni', 0)):
        args.nni_params = nni.get_next_parameter()
    else:
        args.nni_params = None

    args.config_parser = cfg
    return args
