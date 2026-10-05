import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audiovisual.train_loop import train_task
from common.utils import repo_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio-visual binary detector (task 2)")
    parser.add_argument("--config", default=repo_file("configs", "default.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--gpus", default="all")
    args = parser.parse_args()
    train_task("audiovisual", args)
