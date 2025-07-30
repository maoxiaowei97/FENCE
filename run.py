import argparse
import torch
import os
import configparser
from dataset_traffic import get_dataloader
from main_model import CSDI_Traffic
from utils import train, evaluate
import nni

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="config/PEMS08.conf")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='"train" 或 "eval"')
parser.add_argument('--miss_rate',type=str,default=None,help = 'Miss Rate')

parser.add_argument('--guidance', type=str, default=None, choices=['cfg', 'fbg'], help='指定指导方法 (cfg or fbg)')
parser.add_argument('--cfg_scale', type=float, default=None, help='(仅评估时) 指定CFG Scale')
parser.add_argument('--fbg_mode', type=str, default=None, choices=['global', 'spatial', 'cluster'], help='(仅评估时) 指定FBG模式')
parser.add_argument("--results_file", type=str, default=None, help="将评估结果保存到的CSV文件路径")

parser.add_argument("--savename",type=str,default="", help="训练模式下模型保存的基础名称")
parser.add_argument("--load_savename", type=str, default=None, 
                    help="(评估模式下) 要加载的模型的基础名称。例如 'pems08_miss_0.5'")

parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)

args = parser.parse_args()
print(args)

config = configparser.ConfigParser()
config.read(args.config)

# Cover
config["model"]["target_strategy"] = args.targetstrategy
if args.miss_rate is not None:
    config["train"]["miss_rate"] = args.miss_rate
if args.guidance is not None:
    config["model"]["guidance"] = args.guidance
if args.cfg_scale is not None:
    config["model"]["cfg_scale"] = str(args.cfg_scale)
if args.fbg_mode is not None:
    if "fbg" not in config:
        config.add_section("fbg")
    config["fbg"]["mode"] = args.fbg_mode

print("miss rate:{}".format(config["train"]["miss_rate"]))
print("guidance:{}".format(config["model"]["guidance"]))

data_prefix = config['file']['data_prefix']
val_ratio = float(config['train']['val_ratio'])
test_ratio = float(config['train']['test_ratio'])
use_nni = int(config['train']['use_nni'])
sample_len = int (config['train']['sample_len'])
batch_size = int (config['train']['batch_size'])
nsample = int(config['diffusion']['nsample']) #diffusion评估时采样次数

if use_nni:
    params = nni.get_next_parameter()

    target_strategy = params['target_strategy']
    config["model"]["target_strategy"] = target_strategy

    timeemb = int(params['timeemb'])
    config['model']['timeemb'] = str(timeemb)

    featureemb = int(params['featureemb'])
    config['model']['featureemb'] = str(featureemb)

    layers = int(params['layers'])
    config['diffusion']['layers'] = str(layers)

    diffusion_embedding_dim = int(params['diffusion_embedding_dim'])
    config['diffusion']['diffusion_embedding_dim'] = str(diffusion_embedding_dim)

    nheads = int(params['nheads'])
    config['diffusion']['nheads'] = str(nheads)

true_datapath = os.path.join(data_prefix,f"true_data_{config['train']['type']}_{config['train']['miss_rate']}_v2.npz")
miss_datapath = os.path.join(data_prefix,f"miss_data_{config['train']['type']}_{config['train']['miss_rate']}_v2.npz")
train_loader, valid_loader, test_loader,target_dim,_std,_mean = get_dataloader(
    true_datapath,miss_datapath,val_ratio,test_ratio,batch_size,sample_len
)
model = CSDI_Traffic(config, target_dim,args.device).to(args.device)

if args.mode == 'train':
    print("训练：缺失率 {}...".format(config["train"]["miss_rate"]))
    # 训练模型
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        savename = args.savename
    )
    evaluate(
        model,
        test_loader,
        _std, _mean, use_nni,
        nsample=nsample,
        results_file = args.results_file
    )

elif args.mode == 'eval':
    # if not os.path.exists(args.modelpath):
    #     print(f"错误: 找不到模型文件 {args.modelpath}")
    #     exit(1)
    #
    # if args.load_savename is None:
    #     print("错误: 在评估模式下，必须通过 --load_savename 参数指定要加载的模型基础名称。")
    #     exit(1)

    base_path = "./params" # 默认模型保存目录
    # cond_model_path = os.path.join(base_path, f"{args.load_savename}_cond.pth")
    # uncond_model_path = os.path.join(base_path, f"{args.load_savename}_uncond.pth")
    # cond_model_path = '/data/maodawei/USTIN/imputation_benchmark-main/CFG_Depart/params/pems08_100_100_SCTC_depart_0.9_cond.pth'
    # uncond_model_path = '/data/maodawei/USTIN/imputation_benchmark-main/CFG_Depart/params/pems08_100_100_SCTC_depart_0.9_uncond.pth'
    cond_model_path = '/data/maodawei/USTIN/imputation_benchmark-main/CFG_Depart/params/pems08_50_180_SCTC_depart_0.9_cond.pth'
    uncond_model_path = '/data/maodawei/USTIN/imputation_benchmark-main/CFG_Depart/params/pems08_50_180_SCTC_depart_0.9_uncond.pth'

    if not os.path.exists(cond_model_path):
        print(f"错误: 找不到有条件模型文件 {cond_model_path}")
        exit(1)
    if not os.path.exists(uncond_model_path):
        print(f"错误: 找不到无条件模型文件 {uncond_model_path}")
        exit(1)
        
    print(f"Loading conditional model from: {cond_model_path}")
    model.load_state_dict(torch.load(cond_model_path, map_location=args.device))
    
    print(f"Loading unconditional model weights from: {uncond_model_path}")
    uncond_weights = torch.load(uncond_model_path, map_location=args.device)
    model.diffmodel_uncond.load_state_dict(uncond_weights)
    
    # 确保两个模型都处于评估模式
    model.diffmodel_cond.eval()
    model.diffmodel_uncond.eval()
    # --- 修改结束 ---

    evaluate(
        model,
        test_loader,
        _std, _mean, use_nni,
        nsample=nsample,
        results_file=args.results_file
    )