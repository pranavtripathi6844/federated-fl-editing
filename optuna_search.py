import argparse
import optuna
import subprocess
import os
from joblib import Parallel, delayed

# Parameter ranges based on your 15-job results (adjust as needed)
IID_LR = [0.01, 0.005, 0.001]
IID_CLIENT_FRAC = [0.1, 0.3]
IID_LOCAL_EPOCHS = [4, 8, 16]
IID_BATCH_SIZE = [256, 512, 1024]  # Increased batch size for better GPU utilization

NONIID_LR = [0.01, 0.005, 0.001]
NONIID_CLIENT_FRAC = [0.1, 0.3]
NONIID_LOCAL_EPOCHS = [4, 8, 16]
NONIID_BATCH_SIZE = [256, 512, 1024]
NONIID_NUM_CLASSES = [1, 5, 10, 50]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (set to 2 for 2 GPUs)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU index to use for this process (optional)')
    return parser.parse_args()

def objective(trial, gpu=None):
    # Unified search space
    data_distribution = trial.suggest_categorical('data_distribution', ['iid', 'non_iid'])
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.005, 0.001])
    client_fraction = trial.suggest_categorical('client_fraction', [0.1, 0.3, 0.5])
    local_epochs = trial.suggest_categorical('local_epochs', [4, 8, 16])
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])

    log_dir = f"./logs/optuna_{data_distribution}_trial{trial.number}"
    cmd = [
        "python", "train_federated.py",
        "--model_type", "vit_small",
        "--image_size", "128",
        "--batch_size", str(batch_size),
        "--num_clients", "100",
        "--client_fraction", str(client_fraction),
        "--num_rounds", "50",
        "--local_epochs", str(local_epochs),
        "--learning_rate", str(learning_rate),
        "--data_distribution", data_distribution,
        "--log_dir", log_dir,
        "--force_reset"
    ]
    if data_distribution == 'non_iid':
        num_classes_per_client = trial.suggest_categorical('num_classes_per_client', [1, 5, 10])
        cmd += ["--num_classes_per_client", str(num_classes_per_client)]

    # Set CUDA_VISIBLE_DEVICES for subprocess if gpu is specified
    env = os.environ.copy()
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Intermediate pruning (optional, if you have intermediate_acc.txt logic)
    intermediate_file = os.path.join(log_dir, "intermediate_acc.txt")
    if os.path.exists(intermediate_file):
        try:
            with open(intermediate_file, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    try:
                        acc = float(line.strip())
                        trial.report(acc, step=i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    except Exception:
                        continue
        except Exception as e:
            print(f"Could not read intermediate_acc.txt for trial {trial.number}: {e}")

    # Read best accuracy from file
    best_acc = 0.0
    try:
        with open(f"{log_dir}/best_acc.txt") as f:
            best_acc = float(f.read())
    except Exception as e:
        print(f"Could not read best_acc.txt for trial {trial.number}: {e}")
    return best_acc

if __name__ == "__main__":
    args = parse_args()
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", study_name="fl_iid_noniid_search", pruner=pruner)
    # If n_jobs > 1, run in parallel (e.g., n_jobs=2 for 2 GPUs)
    if args.n_jobs > 1:
        # Assign half the trials to each GPU
        def wrapped_objective(trial):
            # Assign GPU 0 to first half, GPU 1 to second half
            gpu = 0 if trial.number < args.n_trials // 2 else 1
            return objective(trial, gpu=gpu)
        study.optimize(wrapped_objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    else:
        # If --gpu is specified, use it for all trials in this process
        def wrapped_objective(trial):
            return objective(trial, gpu=args.gpu)
        study.optimize(wrapped_objective, n_trials=args.n_trials)
    print("Best trial:")
    print(study.best_trial) 