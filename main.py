import  os
import pickle
import logging
import numpy as np

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My libraries
from config import load_config
from config.utils import *
from fedlearning import compressor_registry
from fedlearning.myoptimizer import *
from deeplearning import nn_registry
from deeplearning.validate import *
from deeplearning.dataset import *

def train(model, config, logger, record):
    """Simulate Federated Learning training process. 
    
    Args:
        model (nn.Module):       the model to be trained.
        config (class):          the user defined configuration.
        logger (logging.logger): a logger for train info output.
        record (dict):           a record for train info saving.  
    """    
    # initialize userIDs
    users_to_sample = config.users
    user_ids = np.arange(0, config.users)

    # initialize local residual compensation term
    local_residuals = {}
    for user_id in user_ids:
        local_residuals[user_id] = GradBuffer(model.state_dict(), mode="zeros")

    # initialize the optimizer for the server model
    dataset = assign_user_data(config, logger)

    global_updater = GlobalUpdater(config, model.state_dict()) 

    # before optimization, report the result first
    validate_and_log(model, global_updater, dataset, config, record, logger)
    
    for comm_round in range(config.rounds):
        userIDs_candidates = user_ids
        
        # Wait for all users updating locally
        local_packages = []
        for i, user_id in enumerate(userIDs_candidates):
            user_resource = assign_user_resource(config, user_id, 
                                dataset["train_data"], dataset["user_with_data"])
            updater = LocalUpdater(user_resource, config)
            updater.local_step(model, local_residuals[user_id].grad_dict())
            local_package = updater.uplink_transmit(local_residuals[user_id].grad_dict())
            local_packages.append(local_package)

        # for i, userID in enumerate(attacker_list):
        #     local_package = random_grad_package(model)
        #     local_packages.append(local_package)

        # Update the global model
        global_updater.global_step(model, local_packages)

        # log and record
        logger.info("Round {:d}".format(comm_round))
        validate_and_log(model, global_updater, dataset, config, record, logger)

        if comm_round == config.scheduler[0]:
            config.lr *= config.lr_scaler
            config.scheduler.pop(0)

def main():
    config = load_config()
    logger = init_logger(config)

    user_with_datas = [
        "/media/kaiyue/2D8A97B87FB4A806/Datasets/user_with_data/fmnist/iid/iid_mapping_0.dat",
        # "/media/kaiyue/2D8A97B87FB4A806/Datasets/user_with_data/fmnist/iid/iid_mapping_1.dat",
        # "/media/kaiyue/2D8A97B87FB4A806/Datasets/user_with_data/fmnist/iid/iid_mapping_2.dat",
        # "/media/kaiyue/2D8A97B87FB4A806/Datasets/user_with_data/fmnist/a0.5/user_dataidx_map_0.50_0.dat",
        # "/media/kaiyue/2D8A97B87FB4A806/Datasets/user_with_data/fmnist/a0.5/user_dataidx_map_0.50_1.dat",
        # "/media/kaiyue/2D8A97B87FB4A806/Datasets/user_with_data/fmnist/a0.5/user_dataidx_map_0.50_2.dat",
    ]

    for user_with_data in user_with_datas:
        config.user_with_data = user_with_data

        model = init_model(config, logger)
        record = init_record(config, model)
        train(model, config, logger, record)
        save_record(config, record)

if __name__ == "__main__":
    main()