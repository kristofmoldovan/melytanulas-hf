# Deep Learning Class (VITMMA19) Project Work template

[Complete the missing parts and delete the instruction parts before uploading.]

## Submission Instructions

[Delete this entire section after reading and following the instructions.]

### Project Levels

**Basic Level (for signature)**
*   Containerization
*   Data acquisition and analysis
*   Data preparation
*   Baseline (reference) model
*   Model development
*   Basic evaluation

**Outstanding Level (aiming for +1 mark)**
*   Containerization
*   Data acquisition and analysis
*   Data cleansing and preparation
*   Defining evaluation criteria
*   Baseline (reference) model
*   Incremental model development
*   Advanced evaluation
*   ML as a service (backend) with GUI frontend
*   Creative ideas, well-developed solutions, and exceptional performance can also earn an extra grade (+1 mark).

### Data Preparation

**Important:** You must provide a script (or at least a precise description) of how to convert the raw database into a format that can be processed by the scripts.
* The scripts should ideally download the data from there or process it directly from the current sharepoint location.
* Or if you do partly manual preparation, then it is recommended to upload the prepared data format to a shared folder and access from there.

[Describe the data preparation process here]

### Logging Requirements

The training process must produce a log file that captures the following essential information for grading:

1.  **Configuration**: Print the hyperparameters used (e.g., number of epochs, batch size, learning rate).
2.  **Data Processing**: Confirm successful data loading and preprocessing steps.
3.  **Model Architecture**: A summary of the model structure with the number of parameters (trainable and non-trainable).
4.  **Training Progress**: Log the loss and accuracy (or other relevant metrics) for each epoch.
5.  **Validation**: Log validation metrics at the end of each epoch or at specified intervals.
6.  **Final Evaluation**: Result of the evaluation on the test set (e.g., final accuracy, MAE, F1-score, confusion matrix).

The log file must be uploaded to `log/run.log` to the repository. The logs must be easy to understand and self explanatory. 
Ensure that `src/utils.py` is used to configure the logger so that output is directed to stdout (which Docker captures).

### Submission Checklist

Before submitting your project, ensure you have completed the following steps.
**Please note that the submission can only be accepted if these minimum requirements are met.**

- [X] **Project Information**: Filled out the "Project Information" section (Topic, Name, Extra Credit).
- [ ] **Solution Description**: Provided a clear description of your solution, model, and methodology.
- [X] **Extra Credit**: If aiming for +1 mark, filled out the justification section.
- [X] **Data Preparation**: Included a script or precise description for data preparation.
- [X] **Dependencies**: Updated `requirements.txt` with all necessary packages and specific versions.
- [X] **Configuration**: Used `src/config.py` for hyperparameters and paths, contains at least the number of epochs configuration variable.
- [ ] **Logging**:
    - [ ] Log uploaded to `log/run.log`
    - [ ] Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.
- [ ] **Docker**:
    - [X] `Dockerfile` is adapted to your project needs.
    - [X] Image builds successfully (`docker build -t dl-project .`).
    - [X] Container runs successfully with data mounted (`docker run ...`).
    - [X] The container executes the full pipeline (preprocessing, training, evaluation).
- [ ] **Cleanup**:
    - [X] Removed unused files.
    - [ ] **Deleted this "Submission Instructions" section from the README.**

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
`data` -> `/app/data` - Data folder in the below described format
`inference` -> `/app/inference`- The folder including a timeseries which will be classified on a moving window by the model. The path of this .CSV must be set in the config.py file. (It is already set for the repository folder by default)
`output` -> `/app/output` - The folder will contain these in the end of process: models, model training logs, plots, prediction of inference CSV file
`src` -> `/app/src`
`notebook` -> `/app/notebook`

**Important!** The data folder must be in this format:
- **`data/`**: Contains the source code for the machine learning pipeline.
    - `raw`: Folder containing the NEPTUN folders
      - `NEPTUN1`: Example NEPTUN folder containing a json and multiple csv files
        - `.json`: Obe json containing the labels
        - `.csv`: Multiple .CSV files containing the prices that were labeled in the json

A `processed` folder will be created next to the `raw` folder when data processing has ran.

**To capture the logs for submission (required), redirect the output to a file:**

Assuming that PWD is the repository folder, and data folder is in the right format:

```bash
docker run --rm --gpus all \
  -v "${PWD}/data":/app/data \
  -v "${PWD}/inference":/app/inference \
  -v "${PWD}/output":/app/output \
  -v "${PWD}/src":/app/src \
  -v "${PWD}/notebook":/app/notebook \
  dl-project > log/run.log 2>&1
```

*   Replace `/absolute/path/to/your/local/data` with a path to a **PARENT FOLDER** containing your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `00-print-config.py`: Script to print configuration variables and model architectures for verification.
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

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `run.sh`: Running the whole perparation-training-evaluation-inference pipeline
