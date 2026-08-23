# Numerical Experiments for Finding Ellipses

This repository contains numerical experiments for studying the properties of matrices related to the "Finding Ellipses" problem. The experiments focus on analyzing the structure of these matrices and training neural networks to classify them based on their properties.

## Repository Structure

- `matrixgen/`: Contains scripts for generating matrices and extracting features.
- `neuralnet/`: Contains scripts for training and evaluating neural networks.
- `data/`: Stores generated datasets.
- `checkpoints/`: Stores trained neural network models.

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: `numpy`, `torch`, `sklearn`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Generating Data

First, generate the dataset of matrices:
```bash
python matrixgen/generate_dataset.py
```

This will create `data/dataset_arrays.npz` containing training and validation data.

### Training Neural Networks

Train the hierarchical classification models:
```bash
python neuralnet/train_hierarchical_model.py
```

This will train three separate models for different classification tasks and save the checkpoints in `checkpoints/`.

### Evaluating Models

Evaluate the trained models on the validation set:
```bash
python neuralnet/hierarchical_classification.py
```

This will print classification reports and confusion matrices for each level of the hierarchy.

## License

This project is licensed under the terms of the MIT license.