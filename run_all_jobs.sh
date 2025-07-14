#!/bin/bash

# List all your 15 commands here, in the order you want them to run
commands=(
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 4 --data_distribution iid --num_rounds 50 --log_dir ./logs/fl_iid_j4_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 8 --data_distribution iid --num_rounds 50 --log_dir ./logs/fl_iid_j8_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 16 --data_distribution iid --num_rounds 50 --log_dir ./logs/fl_iid_j16_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 4 --data_distribution non_iid --num_classes_per_client 1 --num_rounds 50 --log_dir ./logs/fl_noniid_nc1_j4_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 4 --data_distribution non_iid --num_classes_per_client 5 --num_rounds 50 --log_dir ./logs/fl_noniid_nc5_j4_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 4 --data_distribution non_iid --num_classes_per_client 10 --num_rounds 50 --log_dir ./logs/fl_noniid_nc10_j4_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 4 --data_distribution non_iid --num_classes_per_client 50 --num_rounds 50 --log_dir ./logs/fl_noniid_nc50_j4_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 8 --data_distribution non_iid --num_classes_per_client 1 --num_rounds 50 --log_dir ./logs/fl_noniid_nc1_j8_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 8 --data_distribution non_iid --num_classes_per_client 5 --num_rounds 50 --log_dir ./logs/fl_noniid_nc5_j8_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 8 --data_distribution non_iid --num_classes_per_client 10 --num_rounds 50 --log_dir ./logs/fl_noniid_nc10_j8_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 8 --data_distribution non_iid --num_classes_per_client 50 --num_rounds 50 --log_dir ./logs/fl_noniid_nc50_j8_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 16 --data_distribution non_iid --num_classes_per_client 1 --num_rounds 50 --log_dir ./logs/fl_noniid_nc1_j16_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 16 --data_distribution non_iid --num_classes_per_client 5 --num_rounds 50 --log_dir ./logs/fl_noniid_nc5_j16_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 16 --data_distribution non_iid --num_classes_per_client 10 --num_rounds 50 --log_dir ./logs/fl_noniid_nc10_j16_50r --force_reset"
"python train_federated.py --model_type vit_small --num_clients 100 --client_fraction 0.1 --local_epochs 16 --data_distribution non_iid --num_classes_per_client 50 --num_rounds 50 --log_dir ./logs/fl_noniid_nc50_j16_50r --force_reset"
)

# Run jobs strictly sequentially, one at a time
for cmd in "${commands[@]}"; do
  $cmd
done

echo "All jobs finished!"
