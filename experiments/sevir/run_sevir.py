import os
import sys
import random

sys.path.append("../../../")

import yaml
import torch
import argparse
import numpy as np
from typing import Dict
from termcolor import colored
from accelerate import Accelerator
from losses.losses import build_loss_fn
from optimizers.optimizers import build_optimizer
from optimizers.schedulers import build_scheduler
from train_scripts.trainer_ddp import Segmentation_Trainer
from architectures.build_architecture import build_architecture
from dataloaders.sevir.sevir_torch_wrap import SEVIRLightningDataModule


##################################################################################################
def launch_experiment(config_path) -> Dict:
    """
    Builds and launches SegFormer3D experiment on SEVIR dataset.
    Args:
        config (Dict): Configuration file.

    Returns:
        Dict: A dictionary containing model, dataloaders, optimizer, etc.
    """
    
    # load config
    config = load_config(config_path)
    # set seed for reproducibility
    seed_everything(config)

    # build directories
    build_directories(config)

    # 替换为使用 SEVIRLightningDataModule
    data_module = SEVIRLightningDataModule(
        seq_len=config["dataset_parameters"]["seq_len"],
        sample_mode=config["dataset_parameters"]["sample_mode"],
        stride=config["dataset_parameters"]["stride"],
        batch_size=config["dataset_parameters"]["batch_size"],
        layout=config["dataset_parameters"]["layout"],
        output_type=np.float32,
        preprocess=True,
        rescale_method="01",
        verbose=False,
        dataset_name=config["dataset_parameters"]["dataset_name"],
        start_date=config["dataset_parameters"]["start_date"],
        train_val_split_date=config["dataset_parameters"]["train_val_split_date"],
        train_test_split_date=config["dataset_parameters"]["train_test_split_date"],
        end_date=config["dataset_parameters"]["end_date"],
        num_workers=config["dataset_parameters"]["num_workers"]
    )

    # 调用 setup() 初始化数据集
    data_module.prepare_data()  # 准备数据
    data_module.setup(stage=None)  # 确保调用 setup() 完成数据集的分割
    
    # 训练和验证数据加载器
    trainloader = data_module.train_dataloader()
    valloader = data_module.val_dataloader()

    # build the SegFormer3D model
    model = build_architecture(config)

    # set up the loss function
    criterion = build_loss_fn(
        loss_type=config["loss_fn"]["loss_type"],
        loss_args=config["loss_fn"]["loss_args"],
    )

    # set up the optimizer
    optimizer = build_optimizer(
        model=model,
        optimizer_type=config["optimizer"]["optimizer_type"],
        optimizer_args=config["optimizer"]["optimizer_args"],
    )

    # set up schedulers
    warmup_scheduler = build_scheduler(
        optimizer=optimizer, scheduler_type="warmup_scheduler", config=config
    )
    training_scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type="training_scheduler",
        config=config,
    )

    # use Accelerator for distributed training
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=config["training_parameters"][
            "grad_accumulate_steps"
        ],
    )
    accelerator.init_trackers(
        project_name=config["project"],
        config=config,
        init_kwargs={"wandb": config["wandb_parameters"]},
    )

    # display experiment information
    #display_info(config, accelerator, trainset, valset, model)

    # prepare components for accelerator
    model = accelerator.prepare_model(model=model)
    optimizer = accelerator.prepare_optimizer(optimizer=optimizer)
    trainloader = accelerator.prepare_data_loader(data_loader=trainloader)
    valloader = accelerator.prepare_data_loader(data_loader=valloader)
    warmup_scheduler = accelerator.prepare_scheduler(scheduler=warmup_scheduler)
    training_scheduler = accelerator.prepare_scheduler(scheduler=training_scheduler)

    # store the model, dataloaders, optimizer, etc.
    storage = {
        "model": model,
        "trainloader": trainloader,
        "valloader": valloader,
        "criterion": criterion,
        "optimizer": optimizer,
        "warmup_scheduler": warmup_scheduler,
        "training_scheduler": training_scheduler,
    }

    # set up the trainer and run the training process
    trainer = Segmentation_Trainer(
        config=config,
        model=storage["model"],
        optimizer=storage["optimizer"],
        criterion=storage["criterion"],
        train_dataloader=storage["trainloader"],
        val_dataloader=storage["valloader"],
        warmup_scheduler=storage["warmup_scheduler"],
        training_scheduler=storage["training_scheduler"],
        accelerator=accelerator,
    )

    # start training
    trainer.train()


##################################################################################################
def seed_everything(config) -> None:
    """
    Seed everything for reproducibility.
    """
    seed = config["training_parameters"]["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##################################################################################################
def load_config(config_path: str) -> Dict:
    """
    Load the configuration file from the YAML file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


##################################################################################################
def build_directories(config: Dict) -> None:
    """
    Create necessary directories for saving models and logs.
    """
    if not os.path.exists(config["training_parameters"]["checkpoint_save_dir"]):
        os.makedirs(config["training_parameters"]["checkpoint_save_dir"])

    if os.listdir(config["training_parameters"]["checkpoint_save_dir"]):
        raise ValueError("checkpoint exists -- preventing file override -- rename file")


##################################################################################################
def display_info(config, accelerator, trainset, valset, model):
    """
    Display experiment info using the accelerator for distributed training.
    """
    accelerator.print(f"-------------------------------------------------------")
    accelerator.print(f"[info]: Experiment Info")
    accelerator.print(
        f"[info] ----- Project: {colored(config['project'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Group: {colored(config['wandb_parameters']['group'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Name: {colored(config['wandb_parameters']['name'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Batch Size: {colored(config['dataset_parameters']['train_dataloader_args']['batch_size'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Num Epochs: {colored(config['training_parameters']['num_epochs'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Loss: {colored(config['loss_fn']['loss_type'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Optimizer: {colored(config['optimizer']['optimizer_type'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Train Dataset Size: {colored(len(trainset), color='red')}"
    )
    accelerator.print(
        f"[info] ----- Test Dataset Size: {colored(len(valset), color='red')}"
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(
        f"[info] ----- Distributed Training: {colored('True' if torch.cuda.device_count() > 1 else 'False', color='red')}"
    )
    accelerator.print(
        f"[info] ----- Num Classes: {colored(config['model_parameters']['num_classes'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- EMA: {colored(config['ema']['enabled'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Load From Checkpoint: {colored(config['training_parameters']['load_checkpoint']['load_full_checkpoint'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Params: {colored(pytorch_total_params, color='red')}"
    )
    accelerator.print(f"-------------------------------------------------------")


##################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer3D Training Script for SEVIR Dataset.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to yaml config file"
    )
    args = parser.parse_args()
    launch_experiment(args.config)
