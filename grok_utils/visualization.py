import matplotlib.pyplot as plt
import os

def log_metrics(data_dir, operator, train_pct, tr_in_context, val_in_context, lr, epoch, train_acc, val_acc):
    """
    Simple function to log metrics to a file.
    """
    # Make sure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    log_file = os.path.join(data_dir, f"metrics_{operator}.txt")
    
    # If this is the first epoch, write header
    if epoch == 0 and not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("epoch,train_acc,val_acc\n")
    
    # Append metrics for this epoch
    with open(log_file, 'a') as f:
        f.write(f"{epoch},{train_acc},{val_acc}\n")
        
def plot_metrics(data_dir, operator, train_pct, tr_in_context, val_in_context, lr):
    """
    Simple function to plot metrics from log file.
    """
    # Make sure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    log_file = os.path.join(data_dir, f"metrics_{operator}.txt")
    
    if not os.path.exists(log_file):
        print(f"No metrics file found at {log_file}")
        return
    
    # Read metrics
    epochs = []
    train_accs = []
    val_accs = []
    
    with open(log_file, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            epoch, train_acc, val_acc = line.strip().split(',')
            epochs.append(int(epoch))
            train_accs.append(float(train_acc))
            val_accs.append(float(val_acc))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    
    # Add title and labels
    plt.title(f"Training for {operator}: split={train_pct}, tr_context={tr_in_context}, val_context={val_in_context}, lr={lr}")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(data_dir, f"accuracy_plot_{operator}.png"))
    plt.close()
