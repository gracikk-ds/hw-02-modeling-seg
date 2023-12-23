<div align="center">
<p>
<a align="left"  target="_blank">
<img src="logo.png"></a>
</p>
</div>

# PyTorch Lightning Training

This repository contains a script for training models using PyTorch Lightning. The tool facilitates training by setting up logging, callbacks, and hardware configurations. Additionally, it integrates with ClearML for [experiment](https://app.clear.ml/projects/eead857ec4d54b01a846c5a04ef8b7b6/experiments/2d86bd66838e4e1c967a79ff10b5996a/output/execution) tracking.

## Installation

1. Clone the repository:

```bash
git clone ssh://git@gitlab.deepschool.ru:30022/cvr-aug23/a.gordeev/hw-02-modeling-seg.git
```

2. Navigate to the cloned directory:

```bash
cd hw-02-modeling-seg
```

3. Install the required packages:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Usage

To train a model, use the following command:

```bash
dvc pull
python src/cli/train.py --config_path configs/config.yaml

```

## Docker
To use the Docker container for this project, follow these instructions:

1. Build the Docker image:

```bash
docker build -t hw02modeling-seg:latest .
```

2. Run a Docker container:

```bash
docker run --gpus all -it hw02modeling-seg:latest
```

## Features

- **Logging with ClearML**: Automatic logging of experiments to keep track of your training runs.

- **Configurable Callbacks**: The training script provides features like early stopping, learning rate monitoring, and progress bar display using PyTorch Lightning callbacks.

- **Hardware Configuration**: Easily switch between GPU and CPU training with configurable settings.
