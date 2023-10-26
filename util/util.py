import argparse

def parse_args():
    parser = argparse.ArgumentParser("Parse configuration")
    parser.add_argument("--output", type=str, default="output", help="root path of output dir")
    parser.add_argument("--seed", type=int, default=42, help="seed of the random")
    parser.add_argument("--test", default=False, action="store_true", help="whether in test mode")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu id to use. If less than 0, use cpu instead.")
    parser.add_argument("--model_dir", type=str, default="model", help="folder path to save/load neural network models")
    parser.add_argument("--epi", type=int, default=None, help="")
    parser.add_argument("--net", type=str, default="net", help="network name")
    parser.add_argument(
        "--dueling", default=False, action="store_true", help="whether to use dueling dqn (use dqn if false)"
    )

    return parser.parse_args()