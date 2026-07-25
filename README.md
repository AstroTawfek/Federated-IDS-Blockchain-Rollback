# Blockchain-Audited Federated Intrusion Detection System

## Executive Summary

This repository presents a research-focused framework for developing a secure and trustworthy intrusion detection system using federated learning. It is designed to study how distributed intrusion detection can remain robust in the presence of adversarial or poisoned client updates while also preserving transparency through auditable model history.

By combining federated model training, adversarial update simulation, anomaly detection, and provenance-based anchoring, the project provides a practical foundation for secure collaborative learning in cybersecurity applications.

## Motivation

Modern intrusion detection systems increasingly depend on distributed data sources and collaborative model training. However, federated environments are vulnerable to several risks, including:

- malicious or compromised clients,
- label-flipping and data poisoning attacks,
- Byzantine-style model corruption,
- and limited accountability for model updates.

This work addresses those challenges through a layered defense strategy that includes:

- local client training without centralizing raw data,
- detection of suspicious model updates,
- exclusion of unreliable contributions during aggregation,
- and provenance tracking for auditability.

## Project Objectives

The core objectives of this project are to:

1. Build a federated intrusion detection pipeline for network traffic classification.
2. Evaluate the influence of adversarial client behavior on model performance.
3. Detect suspicious client updates using lightweight statistical analysis.
4. Record model provenance in a tamper-resistant and auditable form.
5. Compare centralized and federated learning behavior under realistic threat conditions.

## Technical Approach

The framework supports a complete experimental workflow that includes:

- loading benchmark intrusion detection datasets,
- preprocessing and preparing training tensors,
- partitioning data across simulated clients,
- training local models independently,
- aggregating updates through federated averaging,
- identifying abnormal client behavior,
- and logging provenance for each training round.

The architecture is modular and extensible, making it suitable for further research in secure machine learning.

## Key Features

- Federated training pipeline with configurable client partitions
- Simulation of poisoning scenarios such as label flipping and Byzantine-style corruption
- Statistical anomaly detection for suspicious updates
- Provenance anchoring using Merkle-style hashing and JSONL logging
- Centralized baseline training for comparison
- Reproducible experiments through configuration files
- Structured output generation for models, metrics, and audit trails

## Supported Datasets

The repository is built around two widely used intrusion detection benchmarks:

- NSL-KDD
- CICIDS2017

These datasets provide a strong foundation for evaluating classification performance and robustness under adversarial conditions.

## Repository Structure

```text
.
├── configs/                  # Experiment configuration files
├── data/
│   ├── processed/            # Preprocessed NumPy dataset files
│   └── raw/                  # Raw benchmark CSV files
├── experiments/              # Trained models, metrics, and provenance artifacts
├── src/
│   ├── attacks.py            # Poisoning and corruption attack simulations
│   ├── data.py               # Dataset loading utilities
│   ├── detect.py             # Suspicious update detection logic
│   ├── eda.ipynb             # Exploratory data analysis notebook
│   ├── evaluation.ipynb      # Evaluation and metrics notebook
│   ├── federated.py          # Federated training and evaluation helpers
│   ├── metrics.py            # Metrics and experiment logging
│   ├── model.py              # Neural network architecture
│   ├── preprocess_cicids.ipynb  # CICIDS preprocessing notebook
│   ├── preprocess_nsl.ipynb     # NSL-KDD preprocessing notebook
│   ├── provenance.py         # Provenance anchoring and logging
│   ├── run_experiment.py     # Main federated experiment runner
│   ├── splits.py            # Client split generation
│   └── train_centralized.py  # Centralized baseline training script
└── README.md
```

## Environment Requirements

The project is implemented in Python and requires:

- Python 3.9 or newer
- PyTorch
- NumPy

Install the dependencies with:

```bash
pip install torch numpy
```

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Federated-IDS-Blockchain-Rollback
```

### 2. Prepare the Data

Ensure the required datasets are available in the repository folders:

- preprocessed arrays under `data/processed/`
- raw benchmark files under `data/raw/`

### 3. Run the Federated Experiments

```bash
python src/run_experiment.py
```

This runs the configured experiments for both benchmark datasets using the settings defined in the `configs/` directory.

## Baseline Training

To train a centralized baseline model for comparison:

```bash
python src/train_centralized.py nsl
```

or:

```bash
python src/train_centralized.py cic
```

## Experiment Configuration

Experiment behavior is controlled by JSON configuration files in the `configs/` directory:

- `configs/exp_nsl.json`
- `configs/exp_cic.json`

These files define:

- the number of simulated clients,
- the number of communication rounds,
- the attack type and intensity,
- batch size and training epochs,
- and the detection thresholds for suspicious updates.

## Outputs and Artifacts

The project generates the following outputs in the `experiments/` directory:

- trained model checkpoints,
- round-by-round evaluation metrics,
- provenance records for update auditing,
- and supporting experiment logs.

## Methodology Overview

The workflow can be summarized as follows:

1. Load and preprocess the intrusion detection datasets.
2. Partition the training data among simulated clients.
3. Train local models on each client in a federated round.
4. Apply adversarial update transformations when configured.
5. Detect suspicious updates using a distance-based scoring mechanism.
6. Exclude unreliable updates from aggregation when necessary.
7. Update the global model and record the new provenance anchor.
8. Evaluate the resulting model and store experiment metrics.

## Potential Applications

This framework is relevant to:

- academic research in secure federated learning,
- cybersecurity experiments involving distributed intrusion detection,
- robustness analysis of poisoning and Byzantine attacks,
- and the design of auditable machine learning systems.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project, provided that the original copyright and license notice are included in any copies or substantial portions of the software.

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contribution Guidelines

Contributions are welcome and encouraged. To contribute effectively:

1. Fork the repository and create a feature branch.
2. Make your changes with clear and focused commits.
3. Ensure your code remains compatible with the existing project structure.
4. Add or update documentation when introducing new functionality.
5. Submit a pull request describing the purpose of your changes and the results obtained.

### How to Contribute

- Report bugs or issues through GitHub Issues.
- Suggest improvements or new experiments through pull requests.
- Share ideas for extending the federated detection or provenance modules.
- Help improve reproducibility by documenting setup steps or experiment outcomes.

## Notes

- The implementation uses preprocessed NumPy arrays for efficient experimentation.
- The notebooks in the `src/` directory can be used for exploratory analysis and preprocessing tasks.
- Running the scripts from the repository root is recommended for consistency.

## Conclusion

This repository provides a strong foundation for investigating trustworthy federated intrusion detection systems. It combines practical implementation details with research-oriented experimentation and is suitable for both academic study and further extension into more advanced secure learning systems.
