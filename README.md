## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Kristóf Moldován
- **Aiming for +1 Mark**: Yes

### Solution Description

[Provide a short textual description of the solution here. Explain the problem, the model architecture chosen, the training methodology, and the results.]

### Extra Credit Justification


Main reason: Classification of continous timeseries exported in .CSV format and shown on plot images (Documented in "05-Inference" notebook)


Also added extra features supporting the whole process:
- Incremental model development (Compared 3 minimal models and built a bigger one)
  - Documented in 02-Model-building notebook
- Implemented optional cached Data Loading for fast training (set by default in config.py, uses ~700MB of RAM)
- Implemented prediction on moving window of any continous timeseries coming from the 
  - Documented in 05-Inference notebook
  - The window can be custom sized, so any size of flags can be searched 🙂
- Automatic export of plot images into the output folder during the whole process
- Measured Bullish vs. Bearish capabilities (Examined in "04-Eval-Explaination" notebook)
  - I was assuming the model can differentiate well between "Bearish" and "Bullish" main groups, and the low scores are coming from classifying the subgroups. (Normal, Pennant, Wedge). I examined what if there was only 2 main groups.
  - Turned out this way the model performs better, the source of the original low scores is coming from "too similar" data in Normal, Pennant and Wedge groups, while the Bearish <-> Bullish feature is well recognized.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount the directory to `/app/data` inside the container.

Also mount the following volumes from the root of the repository:
- `data` -> `/app/data` - Data folder in the below described format
- `inference` -> `/app/inference`- The folder including a timeseries which will be classified on a moving window by the model. The path of - this .CSV must be set in the config.py file. (It is already set for the repository folder by default)
- `output` -> `/app/output` - The folder will contain these in the end of process: models, model training logs, plots, prediction of inference CSV file
- `src` -> `/app/src`
- `notebook` -> `/app/notebook`

#### Data Preparation

**Important!**

**The script handles** the data download, but for manual download, the data folder must be in this format:
- **`data/`**: Contains the source code for the machine learning pipeline.
    - `raw`: Folder containing the NEPTUN folders
      - `NEPTUN1`: Example NEPTUN folder containing a json and multiple csv files
        - `.json`: Obe json containing the labels
        - `.csv`: Multiple .CSV files containing the prices that were labeled in the json

A `processed` folder will be created next to the `raw` folder when data processing is done.

**To capture the logs for submission (required), redirect the output to a file:**

Assuming that PWD is the repository folder, and data folder is in the right format, then:


On Linux:

```bash
docker run --rm --gpus all \
  -v "${PWD}/data":/app/data \
  -v "${PWD}/inference":/app/inference \
  -v "${PWD}/output":/app/output \
  -v "${PWD}/src":/app/src \
  -v "${PWD}/notebook":/app/notebook \
  dl-project > log/run.log 2>&1
```

For PowerShell:

```
docker run --rm --gpus all `
   -v "${PWD}/data:/app/data" `
   -v "${PWD}/inference:/app/inference" `
   -v "${PWD}/output:/app/output" `
   -v "${PWD}/src:/app/src" `
   -v "${PWD}/notebook:/app/notebook" `
   dl-project *> .\log\run.log
```

*   Replace `${PWD}/data"` with a path to a **PARENT FOLDER** containing your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   Replace all the other listed volumes mounting the folders in the repository.
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log` on Linux. (On windows `*>` is used.)
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).




### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `00a-print-config.py`: Script to print configuration variables and model architectures for verification.
    - `00b-data-download.py`: Script to download and setup raw data from Google Drive.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.
    - **`lib/`**: Contains modularized components used by the main scripts.
        - `baseline_model.py`: Defines the baseline model architecture.
        - `dataloader.py`: Custom PyTorch Dataset implementation for loading flag data.
        - `model.py`: Defines the main FlagClassifier model architecture.
        - `seed_everything.py`: Utility for setting random seeds for reproducibility.
        - `training_loop.py`: Implements the training and validation loop with early stopping.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-Data-Preprocessing.ipynb`: Notebook for initial exploratory data analysis (EDA) and preprocessing.
    - `02-Model-building.ipynb`: Notebook for incremental model testing and building
    - `03-Training.ipynb`: Notebook for training the Baseline and the Built model
    - `04-Eval-Explaination.ipynb`: Notebook for measuring model metrics and comparing 
    - `05-Inference.ipynb`: Notebook for making predictions on "timestamp, close" csv data, demonstrating custom window size and visualizing the results

- **`output/`**: Contains the output of a RUN ID set in config.py (plot images, model weights, histories and predictions)
    - `submission`: Contains of the output of the last run before the submission.
    - `latest`: Empty at submission. Will contain the output of next runs if config.py is not changed.

- **`log/`**: Contains logs of the runs
    - `submission.log`: The log of the last run before the submission.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `run.sh`: Running the whole perparation-training-evaluation-inference pipeline
