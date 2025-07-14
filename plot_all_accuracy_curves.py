import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Directory containing all experiment logs
LOG_ROOT = './logs/'

# Map subdirectory names to plot labels (add/adjust as needed)
LABEL_MAP = {
    'centralized_baseline': 'Centralized Baseline',
    'centralized_editing_run1': 'Centralized + Editing 1',
    'centralized_editing_run2': 'Centralized + Editing 2',
    'centralized_editing_run3': 'Centralized + Editing 3',
    'fed_iid_editing_run1': 'FedAvg IID + Editing 1',
    'fed_iid_editing_run2': 'FedAvg IID + Editing 2',
    'fed_noniid_editing_run1': 'FedAvg non-IID + Editing 1',
    'fed_noniid_editing_run2': 'FedAvg non-IID + Editing 2',
    'fl_iid_j4_50r': 'FedAvg IID (J=4)',
    'fl_iid_j8_50r': 'FedAvg IID (J=8)',
    'fl_iid_j16_50r': 'FedAvg IID (J=16)',
    'fl_noniid_nc1_j4_50r': 'FedAvg non-IID Nc=1 (J=4)',
    'fl_noniid_nc5_j4_50r': 'FedAvg non-IID Nc=5 (J=4)',
    'fl_noniid_nc10_j4_50r': 'FedAvg non-IID Nc=10 (J=4)',
    'fl_noniid_nc50_j4_50r': 'FedAvg non-IID Nc=50 (J=4)',
    'fl_noniid_nc1_j8_50r': 'FedAvg non-IID Nc=1 (J=8)',
    'fl_noniid_nc5_j8_50r': 'FedAvg non-IID Nc=5 (J=8)',
    'fl_noniid_nc10_j8_50r': 'FedAvg non-IID Nc=10 (J=8)',
    'fl_noniid_nc50_j8_50r': 'FedAvg non-IID Nc=50 (J=8)',
    'fl_noniid_nc1_j16_50r': 'FedAvg non-IID Nc=1 (J=16)',
    'fl_noniid_nc5_j16_50r': 'FedAvg non-IID Nc=5 (J=16)',
    'fl_noniid_nc10_j16_50r': 'FedAvg non-IID Nc=10 (J=16)',
    'fl_noniid_nc50_j16_50r': 'FedAvg non-IID Nc=50 (J=16)',
}

# Find all subdirectories in LOG_ROOT
subdirs = [d for d in os.listdir(LOG_ROOT) if os.path.isdir(os.path.join(LOG_ROOT, d))]

plt.figure(figsize=(14, 8))
plotted = []
for subdir in subdirs:
    log_dir = os.path.join(LOG_ROOT, subdir)
    event_files = glob.glob(f"{log_dir}/events.out.tfevents.*")
    if not event_files:
        continue
    ea = event_accumulator.EventAccumulator(event_files[0])
    try:
        ea.Reload()
    except Exception as e:
        print(f"Could not load event file in {log_dir}: {e}")
        continue
    # Try both 'Accuracy/test' and 'Accuracy/val'
    try:
        acc = ea.Scalars('Accuracy/test')
    except KeyError:
        try:
            acc = ea.Scalars('Accuracy/val')
        except KeyError:
            print(f"No accuracy scalar found in {log_dir}")
            continue
    steps = [x.step for x in acc]
    values = [x.value for x in acc]
    label = LABEL_MAP.get(subdir, subdir)
    plt.plot(steps, values, label=label)
    plotted.append(label)

plt.xlabel('Epoch / Round')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Curves for All Experiments')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('all_accuracy_curves.png', dpi=300)
plt.show()

print("Plotted experiments:")
for label in plotted:
    print(f"- {label}") 