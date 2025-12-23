"""
Training script for the BESSTIE figurative language project.

This module contains functions for fine‑tuning pre‑trained encoder models on
the BESSTIE dataset.  It leverages utility functions from ``utils.py`` to
load and preprocess the data, compute class weights and evaluation metrics,
and implements a manual training loop using PyTorch.  This avoids the
dependency on the Hugging Face ``Trainer`` and the associated ``accelerate``
library, which may not be available in all environments.

Example usage from the command line::

    # Train a sarcasm detector using local CSV files
    python train.py --model_name roberta-base --task Sarcasm \
        --train_file train.csv --valid_file valid.csv --output_dir ./sarcasm_model

    # Train a sentiment classifier loaded from the Hugging Face hub
    python train.py --model_name bert-base-uncased --task Sentiment \
        --output_dir ./sentiment_model

To reproduce the settings used in the BESSTIE paper, set ``--num_epochs 30``
and perform a grid search over the learning rate values {1e‑5, 2e‑5, 3e‑5}【180316227421938†L563-L571】.
"""

import argparse
from typing import Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from utils import (
    load_besstie_from_csv,
    load_besstie_from_hf,
    prepare_dataset,
    compute_class_weights,
    compute_metrics,
)


def train_binary_model(
    model_name: str,
    task: str,
    output_dir: str,
    train_file: Optional[str] = None,
    valid_file: Optional[str] = None,
    learning_rates: Optional[tuple] = None,
    batch_size: int = 8,
    num_epochs: int = 30,
    weight_decay: float = 0.01,
    seed: int = 42,
    use_class_weights: bool = True,
) -> None:
    """Fine‑tune a pre‑trained encoder model on BESSTIE for a binary task.

    This function implements a grid search over learning rates when
    multiple values are provided.  It closely follows the training
    protocol of the BESSTIE paper: 30 epochs, batch size 8, class‑
    weighted cross‑entropy loss【180316227421938†L563-L575】 and learning rate
    selection over {1e‑5, 2e‑5, 3e‑5}【180316227421938†L563-L571】.  After
    training on each candidate learning rate, the model is evaluated
    on the validation split and the model achieving the highest
    macro‑averaged F1 score is saved to ``output_dir``.

    Parameters
    ----------
    model_name : str
        Name or path of a pre‑trained Hugging Face model.  The paper
        uses "roberta-large" as one of its encoder baselines【180316227421938†L540-L552】.
    task : str
        Either ``"Sentiment"`` or ``"Sarcasm"``.
    output_dir : str
        Directory to save the fine‑tuned model and checkpoints.
    train_file : str, optional
        Path to ``train.csv``.  If ``None``, the dataset is loaded from
        Hugging Face using ``load_besstie_from_hf``.
    valid_file : str, optional
        Path to ``valid.csv``.  Required if ``train_file`` is provided.
    learning_rates : tuple, optional
        A tuple of candidate learning rates.  If more than one value is
        provided, a grid search is performed and the best model is
        selected based on macro F1.  If ``None``, defaults to
        ``(1e-5, 2e-5, 3e-5)``.
    batch_size : int, optional
        Batch size per device (default 8).
    num_epochs : int, optional
        Number of training epochs (default 30 to match the paper【180316227421938†L563-L571】).
    weight_decay : float, optional
        Weight decay applied during optimisation.
    seed : int, optional
        Random seed for reproducibility.
    use_class_weights : bool, optional
        Whether to apply class weights to mitigate imbalance【180316227421938†L573-L575】.

    Returns
    -------
    None
        The function trains models and saves the best one to ``output_dir``.
    """
    # Default candidate learning rates if none provided
    if learning_rates is None:
        # Use the three values specified in the BESSTIE paper
        learning_rates = (1e-5, 2e-5, 3e-5)
    # Load dataset either from CSV files or Hugging Face hub once
    if train_file and valid_file:
        dataset = load_besstie_from_csv(train_file, valid_file, task=task)
    else:
        dataset = load_besstie_from_hf(task=task)
    # Initialise tokenizer once (reused across learning rate runs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Preprocess dataset once
    tokenised_dataset = prepare_dataset(tokenizer, dataset)
    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        labels_arr = np.array(tokenised_dataset["train"]["label"])
        class_weights = compute_class_weights(labels_arr)
    # Prepare data loaders (reuse for all runs)
    tokenised_dataset = tokenised_dataset.remove_columns([
        c for c in tokenised_dataset["train"].column_names if c not in {"input_ids", "attention_mask", "label"}
    ])
    tokenised_dataset["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenised_dataset["validation"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(tokenised_dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(tokenised_dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collator)
    # Try to import tqdm for progress bars.  If unavailable, define a dummy function.
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        def tqdm(iterable, *args, **kwargs):  # type: ignore
            return iterable
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Determine device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Track the best model and its metric
    best_f1_macro = -float("inf")
    best_model = None
    best_lr = None
    # Iterate over candidate learning rates
    for lr in learning_rates:
        print(f"\n*** Training with learning rate {lr}" )
        # Initialise a fresh model for each learning rate
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)
        # Configure optimiser
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Set up loss function (weights on device)
        if class_weights is not None:
            weight_tensor = torch.tensor([
                class_weights.get(0, 1.0),
                class_weights.get(1, 1.0),
            ], dtype=torch.float).to(device)
            loss_fct = CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = CrossEntropyLoss()
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"LR {lr} Epoch {epoch + 1}/{num_epochs} [Training]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                if "label" in batch:
                    labels = batch.pop("label")
                elif "labels" in batch:
                    labels = batch.pop("labels")
                else:
                    raise KeyError(
                        "Neither 'label' nor 'labels' found in batch: keys=" + str(list(batch.keys()))
                    )
                optimizer.zero_grad()
                outputs = model(**batch)
                logits = outputs.logits
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            # Validation
            model.eval()
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"LR {lr} Epoch {epoch + 1}/{num_epochs} [Validation]"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if "label" in batch:
                        val_labels = batch.pop("label")
                    elif "labels" in batch:
                        val_labels = batch.pop("labels")
                    else:
                        raise KeyError(
                            "Neither 'label' nor 'labels' found in batch during evaluation"
                        )
                    outputs = model(**batch)
                    logits_val = outputs.logits
                    all_logits.append(logits_val.cpu().numpy())
                    all_labels.append(val_labels.cpu().numpy())
            all_logits_np = np.concatenate(all_logits, axis=0)
            all_labels_np = np.concatenate(all_labels, axis=0)
            metrics = compute_metrics((all_logits_np, all_labels_np))
            print(
                f"LR {lr} Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - "
                f"Acc: {metrics['accuracy']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, "
                f"F1_micro: {metrics['f1_micro']:.4f}"
            )
        # After full training at this learning rate, evaluate final performance
        # Use the last computed metrics (macro F1)
        f1_macro = metrics["f1_macro"]
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_model = model
            best_lr = lr  # type: ignore
        else:
            # Free memory for the model that's not selected
            del model
            torch.cuda.empty_cache()
    # Save the best model and tokenizer
    import os
    os.makedirs(output_dir, exist_ok=True)
    if best_model is not None:
        best_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Best model (LR={best_lr}) saved to {output_dir} with F1_macro={best_f1_macro:.4f}")
    else:
        # This should not happen; but handle gracefully
        print("No model was trained. Please check the training configuration.")


def main():
    parser = argparse.ArgumentParser(description="Fine‑tune a model for sarcasm or sentiment detection using BESSTIE.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-large",
        help=(
            "Pre‑trained Hugging Face model to fine‑tune (default 'roberta-large' to match the paper"
        ),
    )
    parser.add_argument("--task", type=str, choices=["Sentiment", "Sarcasm"], default="Sarcasm", help="Which task to train on")
    parser.add_argument("--train_file", type=str, help="Path to train.csv (optional)")
    parser.add_argument("--valid_file", type=str, help="Path to valid.csv (optional)")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Directory to save the trained model")
    # Accept one or more learning rates.  When multiple are provided, a grid
    # search is performed and the best model is selected based on macro F1.
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[1e-5, 2e-5, 3e-5],
        help="One or more learning rates to evaluate. Defaults to the three values used in the BESSTIE paper",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of training epochs (default 30 to match the BESSTIE paper)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weights for the loss function")
    args = parser.parse_args()
    # Train the model.  The function handles saving the best model to ``output_dir``.
    train_binary_model(
        model_name=args.model_name,
        task=args.task,
        output_dir=args.output_dir,
        train_file=args.train_file,
        valid_file=args.valid_file,
        learning_rates=tuple(args.learning_rates),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_class_weights=not args.no_class_weights,
    )
    print(f"Training complete. Best model saved to {args.output_dir}")


if __name__ == "__main__":
    main()