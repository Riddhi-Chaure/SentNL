"""
SentNL — CLI Entry Point

Commands:
    python main.py train    <task> [--model primary|secondary]
    python main.py predict  <task> "<text>"
    python main.py compare  <task>
    python main.py list
"""

import argparse
import json
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure project root is on sys.path
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import ConfigManager
from src.common.logger import get_logger

# Task-specific imports (register predictors)
from src.tasks.spam.predictor import SpamPredictor
from src.tasks.spam.trainer import SpamTrainer
from src.tasks.sentiment.predictor import SentimentPredictor
from src.tasks.sentiment.trainer import SentimentTrainer
from src.tasks.topics.predictor import TopicsPredictor
from src.tasks.topics.trainer import TopicsTrainer

from src.registry import register_task, get_predictor_class, list_tasks

logger = get_logger("system", "system.log")

# -- Register all tasks ------------------------------------------------
_config_manager = ConfigManager()

register_task("spam", SpamPredictor)
register_task("sentiment", SentimentPredictor)
register_task("topics", TopicsPredictor)

# Trainer mapping (not in registry — trainers are CLI-only)
_TRAINERS = {
    "spam": SpamTrainer,
    "sentiment": SentimentTrainer,
    "topics": TopicsTrainer,
}


# -- CLI Commands ------------------------------------------------------

def cmd_train(args):
    """Train a model for the specified task."""
    task = args.task
    model_key = args.model

    if task not in _TRAINERS:
        logger.error(f"Unknown task: {task}")
        sys.exit(1)

    config = _config_manager.get_task_config(task)
    trainer_cls = _TRAINERS[task]
    trainer = trainer_cls(config, PROJECT_ROOT, model_key=model_key)

    logger.info(f"Starting training for '{task}' (model: {model_key})")
    metrics = trainer.train()

    print("\n" + "=" * 50)
    print(f"  Training Complete — {task}")
    print("=" * 50)
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['f1_weighted']:.4f}")
    print(f"  Training Time:   {metrics['training_time_seconds']:.2f}s")
    print(f"  Inference Time:  {metrics['inference_time_ms']:.2f}ms")
    print("=" * 50 + "\n")


def cmd_predict(args):
    """Run inference on a single text."""
    task = args.task
    text = args.text

    config = _config_manager.get_task_config(task)
    predictor_cls = get_predictor_class(task)
    predictor = predictor_cls(PROJECT_ROOT, config)

    result = predictor.predict(text)
    print(json.dumps(result, indent=2))


def cmd_compare(args):
    """Train both candidate models for a task and compare."""
    task = args.task

    if task not in _TRAINERS:
        logger.error(f"Unknown task: {task}")
        sys.exit(1)

    config = _config_manager.get_task_config(task)
    trainer_cls = _TRAINERS[task]

    results = {}
    for key in ["primary", "secondary"]:
        model_cfg = config.get("models", {}).get(key)
        if not model_cfg:
            continue
        model_name = model_cfg.get("name", key)
        print(f"\n{'-' * 40}")
        print(f"  Training: {model_name}")
        print(f"{'-' * 40}")

        trainer = trainer_cls(config, PROJECT_ROOT, model_key=key)
        metrics = trainer.train()
        results[model_name] = metrics

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"  Model Comparison — {task}")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Time':>8}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 8}")
    for name, m in results.items():
        print(
            f"  {name:<25} {m['accuracy']:>10.4f} "
            f"{m['macro_f1']:>10.4f} {m['training_time_seconds']:>7.2f}s"
        )
    print("=" * 60 + "\n")


def cmd_list(args):
    """List all registered tasks."""
    tasks = list_tasks()
    print("\nRegistered tasks:")
    for t in tasks:
        config = _config_manager.get_task_config(t)
        display = config.get("task", {}).get("display_name", t)
        desc = config.get("task", {}).get("description", "")
        print(f"  - {display} ({t}) - {desc}")
    print()


# -- Argument Parser ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SentNL — Unified NLP Classification Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("task", choices=list(_TRAINERS.keys()))
    train_parser.add_argument(
        "--model",
        default="primary",
        choices=["primary", "secondary"],
        help="Which model candidate to train (default: primary)",
    )

    # predict
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("task", type=str)
    predict_parser.add_argument("text", type=str)

    # compare
    compare_parser = subparsers.add_parser(
        "compare", help="Train & compare both model candidates"
    )
    compare_parser.add_argument("task", choices=list(_TRAINERS.keys()))

    # list
    subparsers.add_parser("list", help="List registered tasks")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "compare": cmd_compare,
        "list": cmd_list,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
