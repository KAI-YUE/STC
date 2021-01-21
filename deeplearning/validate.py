import numpy as np

# PyTorch Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# My Libraries
from config.utils import parse_dataset_type
from deeplearning.dataset import UserDataset

def validate_and_log(model, global_updater, dataset, config, record, logger):
    dataset_type = parse_dataset_type(config)
    with torch.no_grad():
        model.eval()
        # validate the model and log test accuracy
        loss = train_loss(model, dataset["train_data"], dataset_type, device=config.device)
        test_acc = test_accuracy(model, dataset["test_data"], dataset_type, device=config.device)
        record["testing_accuracy"].append(test_acc)
        record["loss"].append(loss)

        compress_ratio = global_updater.compress_ratio
        if compress_ratio is not None: 
            compressed_data_KB = global_updater.compressed_bits/(1024*8)
            record["cumulated_KB"].append(compressed_data_KB + record["cumulated_KB"][-1])

        logger.info("Test accuracy {:.4f}".format(test_acc))
        logger.info("Train loss {:.4f}".format(loss))

        if compress_ratio is not None:
            logger.info("Compress ratio {:.3f}".format(compress_ratio))
            logger.info("transmitted_KB {:.3f}".format(compressed_data_KB))

        logger.info("")
        model.train()

def test_accuracy(model, test_dataset, dataset_type, device="cuda"):
    with torch.no_grad():
        model.eval()
        dataset = UserDataset(test_dataset["images"], test_dataset["labels"], dataset_type)
        num_samples = test_dataset["labels"].shape[0]
        predicted_labels = np.zeros_like(test_dataset["labels"])
        accuracy = 0

        dividers = 100
        batch_size = int(len(dataset)/dividers)
        testing_data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        for i, samples in enumerate(testing_data_loader):
            results = model(samples["image"].to(device))
            predicted_labels = torch.argmax(results, dim=1).detach().cpu().numpy()
            accuracy += np.sum(predicted_labels == test_dataset["labels"][i*batch_size: (i+1)*batch_size]) / results.shape[0]
        
        accuracy /= dividers
        model.train()

    return accuracy

def train_loss(model, train_dataset, dataset_type, device="cuda"):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        dataset = UserDataset(train_dataset["images"], train_dataset["labels"], dataset_type)
        loss = torch.tensor(0.)

        dividers = 100
        batch_size = int(len(dataset)/dividers)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        counter = 0
        for samples in data_loader:
            results = model(samples["image"].to(device))
            loss += criterion(results, samples["label"].to(device))
        
        loss /= dividers

    return loss.item()