# Arithmetic Model Training


## Installation

Install the package in development mode:

```bash
pip install -e .
```

This will install all required dependencies and make the training script available.

## Training a Model

You can train a model using the command-line script:

```bash
python scripts/train.py --operator add
```

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--operator` | Arithmetic operator to train on (`add`, `subtract`, etc.) | *Required* |
| `--data_dir` | Directory for data and output files | `./data` |
| `--train_pct` | Train/validation split percentage | `0.25` |
| `--tr_in_context` | Number of in-context examples for training | `0` |
| `--val_in_context` | Number of in-context examples for validation | `0` |
| `--lr` | Learning rate | `1e-3` |
| `--warmup_steps` | Scheduler warmup steps | `50` |
| `--epochs` | Number of training epochs | `300` |
| `--seed` | Random seed | `42` |

### Examples

Basic training with default parameters:
```bash
python scripts/train.py --operator add
```

Training with custom parameters:
```bash
python scripts/train.py --operator multiply --epochs 20 --lr 1e-4 --tr_in_context 10
```

## Output Files

The training process produces the following output files in the specified data directory:

1. **Trained Model**: `model_{operator}_final.pt`
2. **Metrics Log**: `metrics_{operator}.txt`
3. **Accuracy Plot**: `accuracy_plot_{operator}.png`

The accuracy plot shows training and validation accuracy over epochs, with the training parameters displayed in the title.

## Project Structure

```
.
├── README.md
├── grok_utils/
│   ├── __init__.py        # Package initialization
│   ├── data.py            # Data handling code
│   ├── training.py        # Training functions
│   └── visualization.py   # Visualization tools
├── scripts/
│   └── train.py           # Command-line training script
└── setup.py               # Package installation configuration
```

## Visualization

The training process automatically logs metrics and generates plots. The plot shows:

- Training accuracy (blue line)
- Validation accuracy (red line)
- Title with operator, split percentage, in-context examples, and learning rate

This provides a clear visualization of the model's learning progress over epochs.
