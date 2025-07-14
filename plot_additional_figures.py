import os
import glob
import json
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

LOG_ROOT = './logs/'

# 1. Centralized training loss and accuracy curves
centralized_dirs = [
    'centralized_baseline',
    'centralized_editing_run1',
    'centralized_editing_run2',
    'centralized_editing_run3',
]
plt.figure(figsize=(12, 6))
for subdir in centralized_dirs:
    log_dir = os.path.join(LOG_ROOT, subdir)
    event_files = glob.glob(f"{log_dir}/events.out.tfevents.*")
    if not event_files:
        continue
    ea = event_accumulator.EventAccumulator(event_files[0])
    try:
        ea.Reload()
    except Exception:
        continue
    # Accuracy
    try:
        acc = ea.Scalars('Accuracy/val')
    except KeyError:
        try:
            acc = ea.Scalars('Accuracy/test')
        except KeyError:
            continue
    steps = [x.step for x in acc]
    values = [x.value for x in acc]
    plt.plot(steps, values, label=f'{subdir} (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Centralized Training Accuracy Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('centralized_accuracy_curves.png', dpi=300)
plt.close()

# Loss curves
plt.figure(figsize=(12, 6))
for subdir in centralized_dirs:
    log_dir = os.path.join(LOG_ROOT, subdir)
    event_files = glob.glob(f"{log_dir}/events.out.tfevents.*")
    if not event_files:
        continue
    ea = event_accumulator.EventAccumulator(event_files[0])
    try:
        ea.Reload()
    except Exception:
        continue
    try:
        loss = ea.Scalars('Loss/val')
    except KeyError:
        try:
            loss = ea.Scalars('Loss/test')
        except KeyError:
            continue
    steps = [x.step for x in loss]
    values = [x.value for x in loss]
    plt.plot(steps, values, label=f'{subdir} (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Centralized Training Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('centralized_loss_curves.png', dpi=300)
plt.close()

print('Saved: centralized_accuracy_curves.png, centralized_loss_curves.png')

# 2. Federated learning accuracy across Nc and J
plt.figure(figsize=(14, 8))
for subdir in os.listdir(LOG_ROOT):
    if subdir.startswith('fl_iid_j') or subdir.startswith('fl_noniid_nc'):
        log_dir = os.path.join(LOG_ROOT, subdir)
        event_files = glob.glob(f"{log_dir}/events.out.tfevents.*")
        if not event_files:
            continue
        ea = event_accumulator.EventAccumulator(event_files[0])
        try:
            ea.Reload()
        except Exception:
            continue
        try:
            acc = ea.Scalars('Accuracy/test')
        except KeyError:
            continue
        steps = [x.step for x in acc]
        values = [x.value for x in acc]
        plt.plot(steps, values, label=subdir)
plt.xlabel('Round')
plt.ylabel('Test Accuracy (%)')
plt.title('Federated Learning Accuracy Curves (Nc, J)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('federated_accuracy_curves.png', dpi=300)
plt.close()
print('Saved: federated_accuracy_curves.png')

# 3. Bar chart: mask calibration strategies (Fed non-IID editing runs)
mask_strategies = []
final_accuracies = []
for subdir in os.listdir(LOG_ROOT):
    if subdir.startswith('fed_noniid_editing_run'):
        log_dir = os.path.join(LOG_ROOT, subdir)
        # Get mask type from config.json
        config_path = os.path.join(log_dir, 'config.json')
        if not os.path.exists(config_path):
            continue
        with open(config_path) as f:
            config = json.load(f)
        mask_type = config.get('mask_type', subdir)
        # Try to get best_acc.txt, else use last accuracy from event file
        best_acc_path = os.path.join(log_dir, 'best_acc.txt')
        if os.path.exists(best_acc_path):
            with open(best_acc_path) as f:
                acc = float(f.read())
        else:
            event_files = glob.glob(f"{log_dir}/events.out.tfevents.*")
            if not event_files:
                continue
            ea = event_accumulator.EventAccumulator(event_files[0])
            try:
                ea.Reload()
                accs = ea.Scalars('Accuracy/test')
                acc = accs[-1].value if accs else 0.0
            except Exception:
                acc = 0.0
        mask_strategies.append(mask_type)
        final_accuracies.append(acc)
plt.figure(figsize=(10, 6))
plt.bar(mask_strategies, final_accuracies, color='skyblue')
plt.ylabel('Final Test Accuracy (%)')
plt.title('Mask Calibration Strategy Comparison (Fed non-IID)')
plt.tight_layout()
plt.savefig('mask_strategy_comparison.png', dpi=300)
plt.close()
print('Saved: mask_strategy_comparison.png') 