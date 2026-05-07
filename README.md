Machine learning framework for predicting coiled-coil oligomeric states using register-specific physicochemical features.

CCOligoPred: Interpretable Coiled-Coil Oligomeric State Predictor

CCOligoPred is a highly accurate, interpretable machine learning framework designed to predict the oligomeric states of coiled-coil domains (CCDs). Unlike opaque deep-learning models, CCOligoPred is driven by a novel Register-Based Feature (RBF) extraction module. This ensures that the predictions are rooted in fundamental biophysical and geometric determinants of CCD heptad registers. The framework provides two primary functionalities:

1\. Full Prediction Pipeline: Classifies CCDs into specific multiclass states (Parallel Dimer, Antiparallel Dimer, Trimer, Tetramer) and highly-calibrated binary states (Trimer vs. Non-Trimer).

2\. Standalone RBF Extraction: A dedicated module to calculate the 2041-dimensional RBF matrix for other computation applications of CCD (such as designing of CCD based nanoparticles)

📂 Framework Architecture \& Repository Structure

The repository is structured as a fully deployable Python package:

CCOligoPred-Predictor/

├── ccoligopred/                 # Core Python package

│   ├── cli.py                   # Command-line interface and framework entry point

│   ├── io/                      # Input data parser module (Excel handler)

│   ├── rbf/                     # Register-Based Feature (RBF) calculation module

│   ├── models/                  # Independent multiclass and binary prediction pathways

│   └── weights/                 # Pre-trained ML weights and ANOVA feature subsets

├── datasets/                    # Curated datasets for training and benchmarking

│   └── Oligopred\_trial\_test.xlsx # Sample input file for testing

├── pyproject.toml               # Package configuration and dependencies

├── README.md                    # Documentation

└── LICENSE                      # Open-source license

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

⚙️ Installation Guide (Virtual Environment)

To prevent conflicts with your existing Python packages, we highly recommend installing CCOligoPred inside an isolated virtual environment using Anaconda.

Step 1: Download the repository

Open your Anaconda Prompt (or terminal) and clone the repository:



git clone https://github.com/YourUsername/CCOligoPred-Predictor.git

cd CCOligoPred-Predictor



Step 2: Create and activate a Conda virtual environment

Create a fresh environment with a compatible Python version (e.g., Python 3.11):



conda create -n ccoligopred\_env python=3.11 -y

conda activate ccoligopred\_env



Step 3: Install the CCOligoPred package

Install the tool in editable mode. This will automatically resolve and install all required dependencies (Pandas, Scikit-learn, XGBoost, LightGBM, Imbalanced-learn, etc.):



pip install -e .

Step 4: Add the environment to Jupyter Notebook (Optional but Recommended)

If you plan to run the tool inside Jupyter Notebook, link your new environment to Jupyter:



pip install ipykernel

python -m ipykernel install --user --name=ccoligopred\_env --display-name "Python (CCOligoPred)"



Note: When you open Jupyter Notebook, make sure to select Python (CCOligoPred) from the Kernel menu.

Usage Guide (Jupyter Notebook)



CCOligoPred is designed to be easily executed directly within Jupyter Notebook cells using the ! command-line prefix.

Input Data Format

Before running the commands, ensure your input Excel file contains at least two specific columns:

•	Sequence: The raw amino acid sequence of the Coiled-Coil Domain.

•	Register: The corresponding heptad registers annotation (e.g., abcdefg).

(A sample file Oligopred\_trial\_test.xlsx is provided in the datasets/ folder).



🟢 Command 1: Full Prediction Pipeline

To run both the multiclass (PD, APD, TRI, TET) and binary (TRI vs. Non-TRI) predictors, use the predict sub-command:



\#Run this inside a Jupyter Notebook cell

Python

!ccoligopred predict -i datasets/Oligopred\_trial\_test.xlsx -o ccoligopred\_full\_results.xlsx



Output: Generates an Excel file containing your original sequences alongside the multiclass predictions, binary predictions, and confidence probabilities.

🔵 Command 2: Standalone RBF Extraction Only



If you only want to calculate the 2041-dimensional Register-Based Features (RBF) without running the machine learning models, use the rbf sub-command:

\# Run this inside a Jupyter Notebook cell



Python

!ccoligopred rbf -i datasets/Oligopred\_trial\_test.xlsx -o standalone\_rbf\_matrix.xlsx



Output: Bypasses the classifiers and strictly generates an Excel file containing your sequences mapped to their 2041 mathematical structural features.





