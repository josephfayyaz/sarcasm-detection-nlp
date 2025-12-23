  


"""
Entry point for the BESSTIE figurative language project.

This script wraps the ``train.py`` and ``inference.py`` modules into a single
command‑line interface using subcommands.  The ``train`` subcommand launches
training of a new model, while the ``predict`` subcommand runs inference on
new texts or a CSV file.   ssss

Example usage::

    # Train a sarcasm detector
    python main.py train --task Sarcasm \
        --train_file train.csv --valid_file valid.csv --output_dir ./sarcasm_model

    # Predict sarcasm labels for a list of sentences
    python main.py predict --checkpoint_dir ./sarcasm_model --text "This is great" "Not impressed"

    # Predict on a CSV file
    python main.py predict --checkpoint_dir ./sarcasm_model --input_file new_data.csv
"""

import argparse
import sys

from train import train_binary_model
from inference import predict_binary


def main():
    parser = argparse.ArgumentParser(description="BESSTIE figurative language detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Fine‑tune a model on BESSTIE")
    train_parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-large",
        help="Pre‑trained model to fine‑tune (default 'roberta-large' to match the paper)",
    )
    train_parser.add_argument("--task", type=str, choices=["Sentiment", "Sarcasm"], default="Sarcasm")
    train_parser.add_argument("--train_file", type=str, default="../dataset/train.csv", help="Path to train.csv (optional)")
    train_parser.add_argument("--valid_file", type=str, default="../dataset/valid.csv", help="Path to valid.csv (optional)")
    train_parser.add_argument("--output_dir", type=str, default="./model_output", help="Output directory for the model")
    # Accept a list of learning rates for grid search (default values from the paper)
    train_parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[1e-5, 2e-5, 3e-5],
        help="One or more learning rates to evaluate (default uses the values from the BESSTIE paper)",
    )
    train_parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    train_parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of training epochs (default 30 to match the paper)",
    )
    train_parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--no_class_weights", action="store_true", help="Disable class weights")

    # Predict subcommand
    pred_parser = subparsers.add_parser("predict", help="Run inference with a fine‑tuned model")
    # The checkpoint directory identifies the saved model and tokenizer.  The base model name is no longer required.
    pred_parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./model_output",
        help="Directory containing the fine‑tuned model (defaults to './model_output').",
    )
    pred_parser.add_argument("--input_file", type=str, help="CSV file with a 'text' column for batch prediction")
    pred_parser.add_argument("--output_file", type=str, help="Output CSV file path for predictions")
    pred_parser.add_argument("--text", type=str, nargs="*", help="One or more texts for single prediction")

    args = parser.parse_args()

    if args.command == "train":
        # The training function handles model saving to the output directory
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
    elif args.command == "predict":
        if args.input_file:
            import pandas as pd
            import os
            if not os.path.exists(args.input_file):
                print(f"Input file {args.input_file} does not exist", file=sys.stderr)
                sys.exit(1)
            df = pd.read_csv(args.input_file)
            if "text" not in df.columns:
                print("Input CSV must contain a 'text' column", file=sys.stderr)
                sys.exit(1)
            texts = df["text"].astype(str).tolist()
            # The inference helper loads both the model and tokenizer from the checkpoint directory,
            # so only ``checkpoint_dir`` and the texts are passed.
            preds = predict_binary(args.checkpoint_dir, texts)
            df["prediction"] = preds
            output_path = args.output_file or os.path.splitext(args.input_file)[0] + "_predictions.csv"
            df.to_csv(output_path, index=False)
            print(f"Predictions written to {output_path}")
        else:
            if not args.text:
                print("Either --input_file or --text must be provided for prediction", file=sys.stderr)
                sys.exit(1)
            preds = predict_binary(args.checkpoint_dir, args.text)
            for t, p in zip(args.text, preds):
                label = "sarcastic/positive" if p == 1 else "non‑sarcastic/negative"
                print(f"{t} -> {label}")


if __name__ == "__main__":
    main()