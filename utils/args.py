import argparse


def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--years", type=str, default="")
    parser.add_argument("--model_name", type=str, default="HealthMamba")
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--feature", type=int, default=4)
    parser.add_argument("--output_dim", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--mc_samples", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--export", action="store_true", default=False)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--adj_path", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--node_num", type=int, default=0)
    parser.add_argument("--d_hid", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_scales", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--static_dim", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    return parser


def print_args(logger, args):
    for k, v in vars(args).items():
        logger.info(f"{k:20s}: {v}")
