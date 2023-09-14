import argparse
import logging
import os
import random
import warnings
from pydoc import locate
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from trainer import trainer_synapse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="your_data_set_path/data/Synapse/train_npz",
    help="root dir for train data",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="your_data_set_path/data/Synapse/test_vol_h5",
    help="root dir for test data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--tf_log_dir", type=str, default="./tf-logs/", help="output dir")
parser.add_argument("--max_iterations", type=int, default=10000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=12, help="batch_size per gpu")
parser.add_argument("--num_workers", type=int, default=0, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=10, help="eval_interval")
parser.add_argument("--model_name", type=str, default="synapse", help="model_name")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network base learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument('--pre_trained', default=None,
                        help='(path of trained _model)load trained model to continue train')

parser.add_argument(
    "--module", default="networks.MAXFormer.MAXFormer", help="The module that you want to load as the network, e.g. networks.MAXFormer.MAXFormer"
)

args = parser.parse_args()


if __name__ == "__main__":
    # setting device on GPU if available, else CPU
    transformer = locate(args.module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 9,
        },
    }

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = transformer(num_classes=args.num_classes).to(device)

    if args.pre_trained is not None:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.output_dir + '/synapse_8366.pth')
        net.load_state_dict(checkpoint)

    trainer = {
        "Synapse": trainer_synapse,
    }
    trainer[dataset_name](args, net, args.output_dir, args.tf_log_dir, device)
