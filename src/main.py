# -*- coding: utf-8 -*-

import os
import json
import torch
import src.trainer as trainer
from transformers import logging
from lightning.pytorch import seed_everything

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
logging.set_verbosity_error()


if __name__ == "__main__":
    with open(os.path.join("../config", "genreMERT_ARL.json"), "r") as fr:
        config = json.load(fr)
    # print("\nTraining config:\n", json.dumps(config, indent=3))

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            config["train_config"]["memory_fraction"]
        )
    torch.set_float32_matmul_precision("high")
    
    nb_classes = config["data_config"]["nb_classes"]
    random_state = config["train_config"]["random_seed"]
    seed_everything(random_state, workers=True)
    save_folder = os.path.join(
        "..", "result", "genreMERT_ARL"
    )
    model_weight = os.path.join(
        "..", "weights", "genreMERT_ARL.ckpt"
    )
    trainer.test_model(
        model_config=config["model_config"],
        train_config=config["train_config"],
        data_config=config["data_config"],
        save_folder=save_folder,
        model_weight=model_weight,
    )
