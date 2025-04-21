import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to sys.path to allow imports from grok_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grok_utils.data import VALID_OPERATORS
from grok_utils.training import train

def setup_experiment_directories(base_dir, operator):
    """Setup directories for the experiment results"""
    # Create main results directory
    results_dir = os.path.join(base_dir, f"results_{VALID_OPERATORS[operator]}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different seeds
    seeds_dir = os.path.join(results_dir, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)
    
    # Create directory for summary results
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create directory for t-SNE visualizations
    tsne_dir = os.path.join(results_dir, "tsne")
    os.makedirs(tsne_dir, exist_ok=True)
    
    return results_dir, seeds_dir, summary_dir, tsne_dir

def plot_average_differences(all_diffs, summary_dir, operator):
    """Plot the average differences between training and validation reaching 95%"""
    
    # Calculate average, min and max across seeds for each in-context count
    in_context_counts = sorted(all_diffs.keys())
    avg_diffs = []
    min_diffs = []
    max_diffs = []
    
    for count in in_context_counts:
        values = all_diffs[count]
        avg_diffs.append(np.mean(values))
        min_diffs.append(np.min(values))
        max_diffs.append(np.max(values))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.errorbar(in_context_counts, avg_diffs, 
                yerr=[np.array(avg_diffs) - np.array(min_diffs), 
                      np.array(max_diffs) - np.array(avg_diffs)],
                fmt='o-', capsize=5, linewidth=2, markersize=8)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Average Step Difference Between Validation and Training Reaching 95% Accuracy\nOperator: {VALID_OPERATORS[operator]}')
    plt.xlabel('Number of In-Context Examples')
    plt.ylabel('Step Difference (Val95 - Train95)')
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add annotations for positive/negative meaning
    plt.figtext(0.15, 0.02, "Positive: Validation reached 95% after Training", 
                ha='left', fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
    plt.figtext(0.65, 0.02, "Negative: Validation reached 95% before Training", 
                ha='left', fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, f"{VALID_OPERATORS[operator]}_avg_diff_plot.png"), dpi=300)
    
    # Save the numerical data
    with open(os.path.join(summary_dir, f"{VALID_OPERATORS[operator]}_avg_diff_data.csv"), 'w') as f:
        f.write("in_context,avg_diff,min_diff,max_diff\n")
        for i, count in enumerate(in_context_counts):
            f.write(f"{count},{avg_diffs[i]},{min_diffs[i]},{max_diffs[i]}\n")
    
    plt.close()

def main():
    """Parse command-line arguments and launch the experiment."""
    parser = argparse.ArgumentParser(description="Run arithmetic GPT experiments with t-SNE visualization")
    
    # Required arguments
    parser.add_argument("--operator", type=str, required=True, choices=list(VALID_OPERATORS.keys()),
                        help="Arithmetic operator to train on (use quotes, e.g. \"+\" or \"*\")")
    # Optional arguments with defaults
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory for data and output files (default: ./data)")
    parser.add_argument("--train_pct", type=float, default=0.25,
                        help="Percentage of data to use for training (default: 0.25)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Warmup steps for scheduler (default: 50)")
    parser.add_argument("--epochs", type=int, default=9000,
                        help="Number of training epochs (default: 9000)")
    parser.add_argument("--seeds", type=str, default="42,69,420",
                        help="Comma-separated list of random seeds (default: 42,69,420)")
    parser.add_argument("--max_in_context", type=int, default=20,
                        help="Maximum number of in-context examples to test (default: 20)")
    parser.add_argument("--min_in_context", type=int, default=0,
                        help="Minimum number of in-context examples to test (default: 0)")
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(seed) for seed in args.seeds.split(',')]
    
    # Setup directories
    results_dir, seeds_dir, summary_dir, tsne_dir = setup_experiment_directories(args.data_dir, args.operator)
    
    # Dictionary to store differences for all seeds and in-context counts
    all_differences = defaultdict(list)
    
    print("=" * 80)
    print(f"Starting experiments for operator: {args.operator} ({VALID_OPERATORS[args.operator]})")
    print(f"In-context examples range: {args.min_in_context} to {args.max_in_context}")
    print(f"Seeds: {seeds}")
    print("=" * 80)
    
    # Loop through all required in-context values
    for in_context in range(args.min_in_context, args.max_in_context + 1):
        print(f"\n{'-' * 60}")
        print(f"RUNNING EXPERIMENTS WITH {in_context} IN-CONTEXT EXAMPLES")
        print(f"{'-' * 60}")
        
        # Run experiment with different seeds
        for seed_idx, seed in enumerate(seeds):
            print(f"\nSeed {seed_idx+1}/{len(seeds)}: {seed}")
            
            # Set all random seeds
            import torch
            import random
            
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # Create seed-specific directory
            seed_dir = os.path.join(seeds_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            # Run training
            print(f"Training with {in_context} in-context examples...")
            
            # Call the modified train function
            difference, model = train(
                train_pct=args.train_pct,
                operator=args.operator,
                data_dir=seed_dir,  # Use seed-specific directory
                tr_in_context=in_context,
                val_in_context=in_context,
                lr=args.lr,
                warmap_steps=args.warmup_steps,
                epochs=args.epochs,
                seed=seed,
                tsne_dir=os.path.join(tsne_dir, f"seed_{seed}")  # Pass t-SNE directory
            )
            
            # Store the difference (can be negative)
            all_differences[in_context].append(difference)
            
            print(f"Completed seed {seed} with difference: {difference}")
            
        # After all seeds for this in_context value are done, report average
        avg_diff = np.mean(all_differences[in_context])
        print(f"\nFor {in_context} in-context examples:")
        print(f"  Individual differences: {all_differences[in_context]}")
        print(f"  Average difference: {avg_diff:.2f}")
    
    # After all experiments, plot the summary
    print("\nGenerating summary plots...")
    plot_average_differences(all_differences, summary_dir, args.operator)
    
    print("\nExperiment completed!")
    print(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    main()
