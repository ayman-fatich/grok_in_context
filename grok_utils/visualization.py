"""
Visualization utilities for tracking and plotting training metrics.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def log_metrics(data_dir, operator, train_pct, tr_in_context, val_in_context, lr, iteration, train_acc, val_acc):
    """
    Function to log metrics to a file.
    Designed for logging at regular global step intervals.
    
    Args:
        data_dir: Directory to save logs
        operator: Arithmetic operator name
        train_pct: Training split percentage
        tr_in_context: Number of in-context examples for training
        val_in_context: Number of in-context examples for validation
        lr: Learning rate
        iteration: Global step number
        train_acc: Training accuracy
        val_acc: Validation accuracy
    """
    # Make sure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    log_file = os.path.join(data_dir, f"metrics_{operator}_tr{tr_in_context}_val{val_in_context}.txt")
    
    # If this is the first entry, write header
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("iteration,train_acc,val_acc\n")
    
    # Ensure values are valid numbers
    try:
        iteration = int(iteration)
        train_acc = float(train_acc)
        val_acc = float(val_acc)
        
        # Append metrics for this iteration
        with open(log_file, 'a') as f:
            f.write(f"{iteration},{train_acc:.4f},{val_acc:.4f}\n")
            
    except (ValueError, TypeError) as e:
        print(f"Error logging metrics: {e}")
        print(f"Values: iteration={iteration}, train_acc={train_acc}, val_acc={val_acc}")

def plot_metrics(data_dir, operator, train_pct, tr_in_context, val_in_context, lr, diff):
    """
    Function to plot metrics from log file with logarithmic x-axis scale.
    Designed for measurements taken every 10 global steps.
    """
    # Make sure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    log_file = os.path.join(data_dir, f"metrics_{operator}_tr{tr_in_context}_val{val_in_context}.txt")
    
    if not os.path.exists(log_file):
        print(f"No metrics file found at {log_file}")
        return
    
    # Read metrics
    iterations = []
    train_accs = []
    val_accs = []
    
    with open(log_file, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            iteration, train_acc, val_acc = line.strip().split(',')
            iterations.append(int(iteration))
            train_accs.append(float(train_acc))
            val_accs.append(float(val_acc))
    
    # Check if we have any data points
    if len(iterations) == 0:
        print("No data points found in the metrics file. Skipping plot creation.")
        return
        
    # Create plot with logarithmic x-axis
    plt.figure(figsize=(12, 7))
    
    # Add small offset to handle log(0) case if needed
    iterations_array = np.array(iterations)
    if len(iterations_array) > 0 and np.min(iterations_array) <= 0:
        iterations_array = iterations_array + 1  # Add 1 to avoid log(0)
    
    plt.semilogx(iterations_array, train_accs, 'b-', linewidth=2, label='Training Accuracy')
    plt.semilogx(iterations_array, val_accs, 'r-', linewidth=2, label='Validation Accuracy')
    
    # Add grid for logarithmic scale (grid lines at each power of 10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add minor grid lines
    plt.grid(True, which="minor", ls="--", alpha=0.1)
    
    # Format x-axis ticks for better readability
    plt.minorticks_on()
    
    # Add title and labels
    plt.title(f"Training for {operator}: split={train_pct}, tr_context={tr_in_context}, val_context={val_in_context}, lr={lr} \n Diff = {diff}")
    plt.xlabel('Global Steps (log scale)')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    
    # Add tight layout for better spacing
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(data_dir, f"metrics_{operator}_tr{tr_in_context}_val{val_in_context}.png"), dpi=300)
    plt.close()

