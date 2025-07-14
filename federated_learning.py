import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import time
import os
import math
import copy
import random
import numpy as np
from collections import OrderedDict

from vision_transformer import vit_small, vit_tiny
from data_utils import create_iid_split, create_non_iid_split, get_data_statistics, print_data_statistics
from model_editing import SparseSGDM, calibrate_gradient_mask_alternative

class FederatedLearning:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(config['log_dir'])
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        print(f"Federated Learning initialized with {config['num_clients']} clients")
        print(f"Using device: {self.device}")

    def create_model(self):
        if self.config['model_type'] == 'vit_tiny':
            model = vit_tiny(patch_size=16, num_classes=100)
        else:
            model = vit_small(patch_size=16, num_classes=100)
        model.to(self.device)
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    def create_client_datasets(self, trainset):
        if self.config['data_distribution'] == 'iid':
            client_datasets = create_iid_split(
                trainset, 
                self.config['num_clients'],
                self.config.get('samples_per_client')
            )
        else:
            client_datasets = create_non_iid_split(
                trainset,
                self.config['num_clients'],
                self.config['num_classes_per_client'],
                self.config.get('samples_per_client')
            )
        stats = get_data_statistics(client_datasets)
        print_data_statistics(stats)
        return client_datasets

    def train_client(self, client_id, model, client_dataset, round_num):
        client_loader = DataLoader(
            client_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=4
        )
        if self.config.get('use_model_editing', False):
            optimizer = SparseSGDM(
                model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9
            )
            if round_num == 0 or (round_num % self.config.get('mask_recalibration_freq', 10) == 0):
                print(f"Calibrating gradient masks for client {client_id}")
                gradient_masks = calibrate_gradient_mask_alternative(
                    model, client_loader, self.device,
                    mask_type=self.config.get('mask_type', 'least_sensitive'),
                    sparsity_ratio=self.config.get('sparsity_ratio', 0.5),
                    num_calibration_rounds=self.config.get('num_calibration_rounds', 3),
                    num_samples_per_round=self.config.get('num_samples_per_round', 1000)
                )
                optimizer.set_gradient_masks(gradient_masks)
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9
            )
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        model.train()
        for local_epoch in range(self.config['local_epochs']):
            running_loss = 0.0
            for inputs, labels in client_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
        avg_loss = running_loss / len(client_loader)
        return model.state_dict(), avg_loss

    def aggregate_models(self, client_models, client_weights=None):
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        aggregated_state = {}
        param_names = list(client_models[0].keys())
        for param_name in param_names:
            aggregated_param = torch.zeros_like(client_models[0][param_name])
            for i, client_model in enumerate(client_models):
                aggregated_param += client_weights[i] * client_model[param_name]
            aggregated_state[param_name] = aggregated_param
        return aggregated_state

    def evaluate_model(self, model, test_loader):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc

    def run_federated_training(self, trainset, valset, testset):
        print("Starting Federated Learning training...")
        global_model = self.create_model()
        client_datasets = self.create_client_datasets(trainset)
        test_loader = DataLoader(testset, batch_size=self.config['batch_size'], num_workers=4)
        best_acc = 0.0
        total_start_time = time.time()
        for round_num in range(self.config['num_rounds']):
            round_start_time = time.time()
            num_participating = max(1, int(self.config['client_fraction'] * self.config['num_clients']))
            participating_clients = random.sample(range(self.config['num_clients']), num_participating)
            print(f"\nRound {round_num + 1}/{self.config['num_rounds']}")
            print(f"Participating clients: {participating_clients}")
            client_models = []
            client_losses = []
            for client_id in participating_clients:
                client_model = copy.deepcopy(global_model)
                client_state, client_loss = self.train_client(
                    client_id, client_model, client_datasets[client_id], round_num
                )
                client_models.append(client_state)
                client_losses.append(client_loss)
                print(f"Client {client_id} trained, loss: {client_loss:.4f}")
            global_state = self.aggregate_models(client_models)
            global_model.load_state_dict(global_state)
            test_loss, test_acc = self.evaluate_model(global_model, test_loader)
            avg_client_loss = np.mean(client_losses)
            self.writer.add_scalar('Loss/client_avg', avg_client_loss, round_num)
            self.writer.add_scalar('Loss/test', test_loss, round_num)
            self.writer.add_scalar('Accuracy/test', test_acc, round_num)
            round_time = time.time() - round_start_time
            print(f"Round {round_num + 1} completed in {round_time/60:.2f} minutes")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(global_model.state_dict(), self.config['best_model_path'])
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
            if (round_num + 1) % self.config.get('checkpoint_freq', 10) == 0:
                torch.save({
                    'round': round_num,
                    'model_state_dict': global_model.state_dict(),
                    'best_acc': best_acc,
                    'config': self.config
                }, self.config['checkpoint_path'])
                print(f"Checkpoint saved at round {round_num + 1}")
        total_time = time.time() - total_start_time
        print(f"\nFederated Learning completed!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best test accuracy: {best_acc:.2f}%")
        self.writer.close()
        return global_model, best_acc 