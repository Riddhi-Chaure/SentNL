# SentNL — Unified NLP Classification Platform

A modular NLP experimentation and inference platform supporting multiple text classification tasks with reusable preprocessing, configurable vectorization, experiment tracking, artifact management, and extensible ML architecture.

---

## Features

- **3 Classification Tasks** — Spam Detection, Sentiment Analysis, Topic Classification
- **Shared Preprocessing** — Centralized NLP pipeline (tokenization, lemmatization, stopword removal)
- **Abstract Base Classes** — Adding a new task requires ~30 lines of code
- **Configurable Vectorization** — TF-IDF with YAML-driven parameters (BERT-ready)
- **Experiment Tracking** — Automatic logging of all training runs with metrics
- **Artifact Management** — Versioned model bundles with metadata for reproducibility
- **Model Comparison** — Train and compare multiple model candidates per task
- **Streamlit UI** — Interactive prediction, probability visualisation, and dashboards
- **CLI Interface** — Train, predict, and compare from the command line

## Architecture

```
SentNL/
├── configs/          # YAML configuration (base + per-task)
├── datasets/         # Raw data (spam, sentiment, topics)
├── models/           # Serialized model artifacts (versioned)
├── reports/          # Evaluation reports + experiments.csv
├── logs/             # System, training, inference logs
├── src/
│   ├── common/       # Shared infrastructure
│   │   ├── base/     # Abstract base classes (loader, trainer, predictor, vectorizer)
│   │   ├── artifacts/# Artifact manager
│   │   ├── config.py # YAML config loader with inheritance
│   │   ├── logger.py # Unified logging
│   │   ├── preprocess.py  # Centralized NLP preprocessing
│   │   └── vectorizer.py  # TF-IDF vectorizer wrapper
│   ├── tasks/        # Task-specific modules (spam, sentiment, topics)
│   └── registry.py   # Task → Predictor registry
├── main.py           # CLI entry point
└── streamlit_app.py  # Streamlit web interface
```

## Quick Start

### 1. Create & Activate Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\venv\Scripts\activate.bat

# Activate (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Datasets
Place your CSV files in the appropriate `datasets/` subdirectory.

### 4. Train All Models
```bash
python main.py train spam
python main.py train sentiment
python main.py train topics
```

### 5. Compare Model Candidates
```bash
python main.py compare spam
```

### 6. Run Inference
```bash
python main.py predict spam "Congratulations! You've won a free iPhone!"
python main.py predict sentiment "This movie was absolutely brilliant."
python main.py predict topics "The stock market rallied today."
```

### 7. Launch Streamlit UI
```bash
streamlit run streamlit_app.py
```

## Experiment Tracking

Every training run automatically logs to `reports/experiments.csv`:
- Timestamp, task, model name, version
- Accuracy, precision, recall, F1 scores
- Training time, inference speed
- Dataset hash for reproducibility

## Adding a New Task

1. Create a data loader (inherits `BaseDataLoader`)
2. Create a trainer (inherits `BaseTrainer`, override `_build_model()`)
3. Create a predictor (inherits `BasePredictor`)
4. Write a YAML config
5. Register in `main.py`

Total: ~30 lines of new code.

## License

MIT
