# SeaTurtleModelingFramework
 
## Table of Contents:
- [Description](#description)
- [Data](#data)
- [Source Code](#source-code)
- [Setup](#setup)
- [Usage](#usage)
- [Contact](#contact)
- [License](#license)

## Description

This project focuses on modeling Mediterranean sea turtles using satellite tracking, environmental variables, and machine learning techniques. The goal is to understand the movement patterns and habitat selection of loggerhead sea turtles (Caretta caretta) in the Mediterranean Sea to support conservation and ecosystem management efforts.

## Data

The input data for this project is organized as follows:

- `input` folder: Contains two subfolders:
  - `presence`: Contains Excel files with the presence data of loggerhead sea turtles, including corresponding features.
  - `absence`: Contains Excel files with the absence data of loggerhead sea turtles, including corresponding features.

## Source Code

The source code for this project is organized as follows:

- `src` folder: Contains the source code files and subfolders:
  - `conf`: Contains configuration files for the project. 
  - `lib`: Contains libraries for analysis, machine learning, and utility functions.
  - `main`: Contains the main scripts for running the modeling and analysis. It includes the following files:
    - `analysis.py`: The script performs a t-test on presence data to compare the means of two groups: coastal and pelagic. The results of the t-test are saved to an excel file.
    - `train.py`: The script trains a classifier using presence and absence data. The classifier used is a Random Forest Classifier and the search method is BayesSearchCV.
    - `run.py`: The main scripts that offer a CLI interface to performs the two experiment (training and analysis)

## Setup

To set up the project, follow these steps:

1. Clone the repository: 
    ```
    git clone https://github.com/your_username/SeaTurtleModelingFramework.git
    ```
2. Navigate to the project directory:
    ```
    cd SeaTurtleModelingFramework`
    ```
3. Create a virtual environment: 
    ```
    python3 -m venv venv
    ```
4. Activate the virtual environment:
   - For Windows:
     ```
     venv\Scripts\activate
     ```
   - For macOS and Linux:
     ```
     source venv/bin/activate
     ```
5. Install the required dependencies:
     ```
     pip install -r requirements.txt
     ```
## Usage

To use this framework, follow these steps:

1. Prepare your input data by organizing it into the `input` folder as described above.

2. Run the main script `run.py` located in the `src/main` folder to perform the modeling and analysis. The script offers a CLI interface with the following options:
- `-p` or `--preprocessing`: Run preprocessing (currently not available).
- `-t` or `--training`: Run training.
- `-a` or `--analysis`: Run analysis.

Example command to run training:
```
python -m src.main.run -t
```
Example command to run analysis:
```
python -m src.main.run -a
```

You can also use multiple options together, such as `-t -a`, to perform both training and analysis.

## Contact

For any questions or inquiries, please contact [Rocco Caccioppoli](mailto:rocco.caccioppoli@cmcc.it) or [Rosalia Maglietta](mailto:rosalia.maglietta@cnr.it).

## License

This project is licensed under the [GPL 3.0 License](LICENSE).
