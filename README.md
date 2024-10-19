# GAN Toolkit

A personal toolkit for Generative Adversarial Networks (GANs) in PyTorch.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
  - [Data Preparation](#data-preparation)
    - [Downloading the Dataset](#downloading-the-dataset)
    - [Using Your Own Dataset](#using-your-own-dataset)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Training DCGAN](#training-dcgan)
  - [Running the Jupyter Notebook](#running-the-jupyter-notebook)
- [Docker](#docker)
  - [Installing Docker](#installing-docker)
  - [Building the Docker Image](#building-the-docker-image)
  - [Running the Docker Container](#running-the-docker-container)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Modular Architecture**: Easily extendable codebase with separate modules for datasets, models, trainers, and utilities.
- **Flexible Data Loading**: Supports various datasets with customizable transformations.
- **Docker Support**: Containerized environment for consistent deployment.
- **Configuration Management**: YAML configuration files for easy hyperparameter tuning and experiment management.
- **Unit Testing**: Comprehensive tests for reliability and robustness.
- **Logging and Visualization**: Integrated logging with TensorBoard support for monitoring training progress.

---

## Getting Started

### Prerequisites

- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.6 or higher
- **PyTorch**: Version 1.9.0 or higher
- **CUDA**: For GPU support (optional)
- **Git**: Version control system
- **Docker**: For containerization (optional)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/gan-toolkit.git
   cd gan-toolkit
   ```
### Setting Up the Virtual Environment

It's recommended to use a virtual environment to manage project dependencies.

## Setting Up the Virtual Environment

It's recommended to use a virtual environment to manage project dependencies and prevent conflicts with other projects on your system.

### Using `venv`

#### Create a Virtual Environment
```bash
# For Linux/macOS
python3 -m venv venv

# For Windows
python -m venv venv
```

#### Upgrade `pip`
```bash
pip install --upgrade pip
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Verify Installation
```bash
pip list
```

This should display all the packages listed in `requirements.txt`.

## Data Preparation

### Downloading the Dataset
We provide scripts to download datasets. By default, the project uses the Cervical Cancer (Kaggle) dataset.

#### Steps:

1. **Set Up Kaggle API Credentials**
    - Sign in to Kaggle.
    - Go to your account settings and select "Create New API Token". This will download a `kaggle.json` file.

2. **Place the `kaggle.json` file in the appropriate directory**:
    - **Linux/macOS**: `~/.kaggle/kaggle.json`
    - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

3. **Ensure the file has the correct permissions**:
    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```

4. **Download the Dataset**:
    Run the data download script:
    ```bash
    python scripts/download_data.py
    ```

*Note*: The dataset will be downloaded to the `data/` directory by default.

### Using Your Own Dataset

If you prefer to use your own dataset:

#### Organize Your Dataset

Place your dataset in the `data/` directory with the following structure:
```bash
data/
└── your_dataset_name/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── class2/
        ├── image3.jpg
        ├── image4.jpg
        └── ...
```

#### Update the Configuration

Edit the `configs/config.yaml` file:
```yaml
data:
  dataset_path: './data/your_dataset_name'
  # ... other configurations
```

## Configuration

All configurable parameters are stored in `configs/config.yaml`. Adjust hyperparameters, paths, and other settings as needed.

### Example `config.yaml`:
```yaml
data:
  dataset_path: './data/cervical-cancer-dataset'
  image_size: 64
  batch_size: 128
  num_workers: 4
  image_channels: 3
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]

training:
  num_epochs: 25
  learning_rate: 0.0002
  batch_size: 128
  latent_dim: 100

model:
  name: 'dcgan'
  latent_dim: 100
  feature_d: 64
  feature_g: 64

sample_dir: 'samples'
model_dir: 'models'
```

## Usage

### Training DCGAN
Run the training script with the desired configuration:
```bash
python src/main.py --config configs/config.yaml
```

### Running the Jupyter Notebook

For an interactive demonstration, use the provided Jupyter notebook:

#### Start Jupyter Notebook
```bash
jupyter notebook
```

#### Open the Notebook

Navigate to `notebooks/gan_example.ipynb` and open it.

#### Adjust Paths

Ensure that the notebook can access the project modules by adjusting the system path if necessary:
```python
import sys
sys.path.append('../')
```

#### Run Cells

Execute the cells sequentially to see the model in action.

## Docker

### Installing Docker

Follow the official Docker installation guide for your operating system:

- **Windows and macOS**: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: [Docker Engine](https://docs.docker.com/engine/install/)

### Building the Docker Image

1. Navigate to the project root:
    ```bash
    cd /path/to/gan-toolkit
    ```

2. Build the Docker image:
    ```bash
    docker build -t gan-toolkit .
    ```

### Running the Docker Container
```bash
docker run --gpus all -v $(pwd):/app gan-toolkit
```

#### Notes:

- `--gpus all`: Enables GPU access inside the container (requires NVIDIA Docker support).
- `-v $(pwd):/app`: Mounts the current directory into the container.

### NVIDIA Docker Support (For GPU Access):

#### Install the NVIDIA Container Toolkit:

1. Set up the repository:
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
        && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
        && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```

2. Install the NVIDIA Container Toolkit:
    ```bash
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```

#### Running the Container with NVIDIA Runtime:
```bash
docker run --runtime=nvidia -v $(pwd):/app gan-toolkit
```

## Testing

We use `pytest` for unit testing.

### Running Tests
```bash
pytest tests/
```

### Adding Tests

Tests are located in the `tests/` directory.

Follow the existing examples to add new tests.

Ensure your tests cover:
- Data loading
- Model initialization
- Training steps
- Utility functions

## Contributing

We welcome contributions! Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

### Contribution Guidelines

1. Fork the repository.
2. Create a branch for your feature (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

### Code Style

- Follow PEP 8 guidelines.
- Use docstrings to document modules, classes, and methods.
- Add comments to explain complex code segments.
- Write unit tests for new features.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

**Your Name**

- Email: georgiospathanasiou@yahoo.com
- GitHub: [Georgios Athanasiou](https://github.com/ga83wuw)

Feel free to reach out for any questions or suggestions!

## Acknowledgments

- Inspired by the PyTorch DCGAN tutorial.
- Datasets provided by Kaggle.

**Disclaimer**: This project is intended for educational purposes. Use it responsibly and adhere to the licensing terms of any datasets or external resources used.







