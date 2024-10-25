# Satellite-Image-To-Map-Translation

## Project Folder Structure

```bash
├── data
│   ├── __init__.py              # Initializes the data module
│   ├── datasets.py              # Handles loading and processing of datasets
│   ├── download_utils.py        # Utility functions for downloading datasets
│
├── model
│   ├── __init__.py              # Initializes the model module
│   ├── discriminator.py         # Defines the Discriminator model
│   ├── generator.py             # Defines the Generator model
│   ├── losses.py                # Loss functions used in the training process
│
├── utils
│   ├── __init__.py              # Initializes the utils module
│   ├── initialize_weights.py    # Utility to initialize model weights
│   ├── logger.py                # Handles logging during training
│   ├── main.py                  # Main script to execute the training and testing pipeline
│   ├── predict.py               # Script for making predictions using the trained model
│
├── requirements.txt             # List of dependencies needed to run the project
├── train.py                     # Main training script

```
## Setup and Installation

To run the project, you first need to install the necessary dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

For running the training and doing inference you can run the below command
```bash
python train.py
```
