#!/usr/bin/env python

import argparse
import os
import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple
from pathlib import Path

from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from data import (
    DEFAULT_DATA_DIR,
    EOS,
    VALID_OPERATORS,
    ArithmeticDataset,
    ArithmeticIterator,
)

DEFAULT_LOG_DIR = "logs"

class GPT2ForArithmetic(LightningModule):
    """
    Wrapper for GPT-2 model for training on arithmetic tasks
    """

    def __init__(self, hparams: Namespace) -> None:
        """
        :param hparams: An argparse.Namespace with parameters
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.prepare_data()
        
        # Configure GPT-2 with custom parameters
        config = GPT2Config(
            vocab_size=len(self.train_dataset.tokenizer),
            n_positions=hparams.max_context_len,
            n_layer=hparams.n_layers,
            n_head=hparams.n_heads,
            n_embd=hparams.d_model,
            resid_pdrop=hparams.dropout,
            embd_pdrop=hparams.dropout,
            attn_pdrop=hparams.dropout,
        )
        
        self.model = GPT2LMHeadModel(config)

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        """
        Defines the hyperparameter arguments
        """
        parser.add_argument("--batchsize", type=float, default=0)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--max_context_len", type=int, default=50)
        parser.add_argument("--math_operator", type=str, default="+")
        parser.add_argument("--train_data_pct", type=float, default=0.5)
        parser.add_argument("--tr_in_context", type=int, default=0,
                           help="Number of in-context examples for training")
        parser.add_argument("--val_in_context", type=int, default=0,
                           help="Number of in-context examples for validation")
        parser.add_argument("--warmup_steps", type=int, default=10)
        parser.add_argument("--max_lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument(
            "--logdir",
            type=str,
            default=DEFAULT_LOG_DIR,
        )
        parser.add_argument(
            "--datadir",
            type=str,
            default=DEFAULT_DATA_DIR,
        )
        return parser

    def prepare_data(self) -> None:
        """
        Loads training and validation data
        """
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,
            operator=self.hparams.math_operator,
            data_dir=self.hparams.datadir,
            tr_in_context=self.hparams.tr_in_context,
            val_in_context=self.hparams.val_in_context
        )

    def train_dataloader(self):
        """
        Creates training data iterator
        """
        device = next(self.model.parameters()).device
        iterator = ArithmeticIterator(
            self.train_dataset,
            device,
            shuffle=True,
        )
        return iterator

    def val_dataloader(self):
        """
        Creates validation data iterator
        """
        device = next(self.model.parameters()).device
        iterator = ArithmeticIterator(
            self.val_dataset,
            device,
            shuffle=False,
        )
        return iterator

    def forward(self, input_ids, labels=None):
        """
        Forward pass
        """
        return self.model(input_ids=input_ids, labels=labels)

    def _calculate_accuracy(self, logits, target):
        """
        Calculate accuracy for the batch, focusing on the right-hand side of equations
        For sequences with multiple equations, we focus on the last equation.
        """
        # Find the token index for '='
        eq_token_index = self.train_dataset.tokenizer.stoi["="]
        
        # Find the position of the LAST '=' in the sequence
        eq_positions = torch.nonzero(target[0, :] == eq_token_index, as_tuple=False)
        if eq_positions.numel() == 0:
            # If no equal sign is found, return 0 accuracy
            return torch.tensor(0.0, device=logits.device)
            
        # Get the last occurrence of the equal sign
        last_eq_position = int(eq_positions[-1].squeeze())
        
        # Only calculate accuracy on right hand side of the last equation
        target_rhs = target[..., last_eq_position + 1:]
        logits_rhs = logits[..., last_eq_position + 1:, :]
        
        # Get the predicted tokens
        preds = torch.argmax(logits_rhs, dim=-1)
        
        # Calculate accuracy
        correct = (preds == target_rhs).float()
        accuracy = torch.mean(correct) * 100
        
        return accuracy

    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        input_ids = batch["text"]
        labels = batch["target"]
        
        outputs = self(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(outputs.logits, labels)
        
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'])
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        input_ids = batch["text"]
        labels = batch["target"]
        
        outputs = self(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(outputs.logits, labels)
        
        # Log metrics for this step
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        
        # No need to return values as they're automatically collected through logging
        
    def on_validation_epoch_end(self):
        """
        Process validation epoch results - runs at the end of the validation epoch
        """
        # Access metrics via self.trainer.callback_metrics
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_accuracy = self.trainer.callback_metrics.get("val_accuracy")
        
        if val_loss is not None:
            # Calculate perplexity
            perplexity = torch.exp(val_loss)
            self.log("val_perplexity", perplexity)

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler
        """
        # Setup optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.max_lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8
        )
        
        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
        
    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch
        Updates the visualization if needed
        """
        if hasattr(self.trainer, 'training_metrics'):
            self.trainer.training_metrics['train_acc'].append(self.trainer.callback_metrics.get('train_accuracy', 0))
            self.trainer.training_metrics['val_acc'].append(self.trainer.callback_metrics.get('val_accuracy_epoch', 0))
            self.trainer.training_metrics['epochs'].append(self.current_epoch)
            
            # Update the visualization every 5 epochs or at the end of training
            if self.current_epoch % 5 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
                plot_training_progress(
                    self.trainer.training_metrics,
                    os.path.join(self.hparams.logdir, "training_progress.png")
                )


def add_args(parser=None) -> Namespace:
    """
    Parses the command line arguments
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser = GPT2ForArithmetic.add_model_specific_args(parser)
    return parser


def plot_training_progress(metrics, save_path):
    """
    Create and save a plot of training and validation accuracy
    
    :param metrics: Dictionary containing training metrics
    :param save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Convert any PyTorch tensors to Python lists
    epochs = [e.item() if hasattr(e, 'item') else float(e) for e in metrics['epochs']]
    train_acc = [a.item() if hasattr(a, 'item') else float(a) for a in metrics['train_acc']]
    val_acc = [a.item() if hasattr(a, 'item') else float(a) for a in metrics['val_acc']]
    
    # Plot training accuracy
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    
    # Plot validation accuracy
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training progress plot saved to {save_path}")

def train(hparams: Namespace) -> str:
    """
    This is the main trainer method. Sets up and runs the experiment
    with the defined hyperparameters.
    
    :param hparams: An argparse.Namespace with hyperparameters
    :return: Path to the log directory
    """
    # Process args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)
    
    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create the output directories
    os.makedirs(hparams.logdir, exist_ok=True)
    checkpoint_path = os.path.join(hparams.logdir, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Create model
    model = GPT2ForArithmetic(hparams)
    
    # Setup logger
    logger = CSVLogger(hparams.logdir)
    
    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    # Setup trainer
    trainer_args = {
        "max_steps": hparams.max_steps,
        "logger": logger,
        "callbacks": [checkpoint_callback],
        "gradient_clip_val": 1.0,
    }
    
    if torch.cuda.is_available() and hparams.gpu >= 0:
        trainer_args["gpus"] = [hparams.gpu]
    
    trainer = Trainer(**trainer_args)
    
    # Initialize metrics tracking for visualization
    trainer.training_metrics = {
        'train_acc': [],
        'val_acc': [],
        'epochs': []
    }
    
    # Train the model
    start_time = time.time()
    try:
        trainer.fit(model)
        training_time = time.time() - start_time
        
        # Save the final model
        final_model_path = os.path.join(hparams.logdir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        model.model.save_pretrained(final_model_path)
        
        # Create a final visualization
        log_path = Path(hparams.logdir)
        metrics_path = log_path / "metrics.csv" 
        
        if metrics_path.exists():
            # If the CSV logger has created a metrics file, use that for more accurate plotting
            import pandas as pd
            metrics_df = pd.read_csv(metrics_path)
            
            if 'train_accuracy' in metrics_df.columns and 'val_accuracy' in metrics_df.columns:
                epochs = metrics_df['epoch'].values.tolist()
                train_acc = metrics_df['train_accuracy'].values.tolist()
                val_acc = metrics_df['val_accuracy'].values.tolist()
                
                final_metrics = {
                    'epochs': epochs,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                }
                
                try:
                    plot_training_progress(
                        final_metrics,
                        os.path.join(hparams.logdir, "final_training_progress.png")
                    )
                except Exception as e:
                    print(f"Warning: Could not create final training progress plot: {e}")
        
        # Print summary information
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Model saved to {final_model_path}")
        print(f"Logs saved to {hparams.logdir}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    return hparams.logdir


def add_test_functionality(parser):
    """
    Add test-specific functionality to the argument parser
    """
    parser.add_argument("--test", action="store_true", help="Run in test mode on validation data")
    parser.add_argument("--test_checkpoint", type=str, help="Path to checkpoint for testing")
    return parser

def main():
    """
    Main function to parse args and run training
    """
    parser = add_args()
    parser = add_test_functionality(parser)
    hparams = parser.parse_args()
    
    if hparams.test:
        # Test mode
        if not hparams.test_checkpoint:
            print("Error: --test_checkpoint required when running in test mode")
            return
        
        # Create the model and load from checkpoint
        model = GPT2ForArithmetic.load_from_checkpoint(hparams.test_checkpoint)
        
        # Setup trainer for test
        trainer = Trainer(gpus=[hparams.gpu] if torch.cuda.is_available() and hparams.gpu >= 0 else None)
        
        # Run test
        test_results = trainer.test(model)
        print("Test Results:", test_results)
    else:
        # Training mode
        train(hparams)


if __name__ == "__main__":
    main()
