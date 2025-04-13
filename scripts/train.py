import os
import argparse
import sys

# Add parent directory to sys.path to allow imports from grok_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grok_utils.data import VALID_OPERATORS
from grok_utils.training import train

def main():
    """Parse command-line arguments and launch training."""
    parser = argparse.ArgumentParser(description="Train an arithmetic GPT model")
    
    # Required arguments
    parser.add_argument("--operator", type=str, required=True, choices=list(VALID_OPERATORS.keys()),
                        help="Arithmetic operator to train on (use quotes, e.g. \"+\" or \"*\")")
    # Optional arguments with defaults
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory for data and output files (default: ./data)")
    parser.add_argument("--train_pct", type=float, default=0.25,
                        help="Percentage of data to use for training (default: 0.25)")
    parser.add_argument("--tr_in_context", type=int, default=0,
                        help="Number of in-context examples for training (default: 0)")
    parser.add_argument("--val_in_context", type=int, default=0,
                        help="Number of in-context examples for validation (default: 0)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Warmup steps for scheduler (default: 50)")
    parser.add_argument("--epochs", type=int, default=9000,
                        help="Number of training epochs (default: 9000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    import torch
    import random
    import numpy as np
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Print training configuration
    print("=" * 50)
    print("Training Configuration:")
    print(f"  Operator:       {args.operator} ({VALID_OPERATORS[args.operator]})")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Train/Val split: {args.train_pct:.2f}/{1-args.train_pct:.2f}")
    print(f"  In-context examples: {args.tr_in_context} (train), {args.val_in_context} (val)")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Warmup steps:   {args.warmup_steps}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Seed:           {args.seed}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device:         {device}")
    print("=" * 50)
    
    # Start training
    print("\nStarting training...\n")
    
    model = train(
        train_pct=args.train_pct,
        operator=args.operator,
        data_dir=args.data_dir,
        tr_in_context=args.tr_in_context,
        val_in_context=args.val_in_context,
        lr=args.lr,
        warmap_steps=args.warmup_steps,
        epochs=args.epochs,
    )
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
