```markdown
# CCOligoPred: Decoding Oligomeric States of Coiled-Coil Domains Using Register-Based Features and Machine Learning Frameworks

CCOligoPred is a robust, interpretable machine learning framework designed to predict the oligomeric states of Coiled-Coil Domains (CCDs). By utilizing Register-Based Features (RBF) and a dual-pathway architecture (Multiclass and Binary), it achieves high accuracy in distinguishing trimeric assemblies and classifying APD, PD, TRI, and TET states. The frame provides two frameworks, i. Full Prediction Pipeline: Classifies CCDs into specific multiclass states (Parallel Dimer, Antiparallel Dimer, Trimer, Tetramer) and highly-calibrated binary states (Trimer vs. Non-Trimer) and ii. Standalone RBF Extraction: A dedicated module to calculate the 2041-dimensional RBF matrix for other computation applications of CCD (such as designing CCD-based nanoparticles). 

---

## 📂 Framework Architecture & Repository Structure

The repository is structured as a fully deployable Python package:

```text
CCOligoPred-Predictor/
├── ccoligopred/                 # Core Python package
│   ├── cli.py                   # Command-line interface and framework entry point
│   ├── io/                      # Input data parser module (Excel handler)
│   ├── rbf/                     # Register-Based Feature (RBF) calculation module
│   ├── models/                  # Independent multiclass and binary prediction pathways
│   └── weights/                 # Pre-trained ML weights and ANOVA feature subsets
├── datasets/                    # Curated datasets for training and benchmarking
│   └── Oligopred_trial_test.xlsx # Sample input file for testing
├── pyproject.toml               # Package configuration and dependencies
├── README.md                    # Documentation
└── LICENSE                      # Open-source license

```

---

## 🛠️ Prerequisites

Before installing, ensure your system has the following installed:

1. **Anaconda** or **Miniconda** (for managing virtual environments).
2. **Git** (for downloading the repository).

---

## 📥 1. Installation Guide (Recommended Workflow)

To ensure perfect reproducibility and avoid any dependency conflicts, please follow this exact installation order.

**Step 1.1: Open your Anaconda Prompt (or Terminal) and create a fresh virtual environment:**

```bash
conda create -n ccoligopred_env python=3.11 -y
conda activate ccoligopred_env

```

**Step 1.2: Clone the repository:**

```bash
git clone [https://github.com/sirfkisalay/CCOligoPred.git](https://github.com/sirfkisalay/CCOligoPred.git)

```

**Step 1.3: Navigate into the folder and install the framework:**
*Note: The `pyproject.toml` file enforces strict package versions (e.g., scikit-learn==1.5.2) to guarantee stability.*

```bash
cd CCOligoPred
pip install -e .

```

*(Don't forget the dot `.` at the end of the command!)*

---

## 📓 2. Jupyter Notebook Setup (Optional but Recommended)

If you prefer running the framework inside a Jupyter Notebook, you must link your new virtual environment to Jupyter.

**Run these commands in your Anaconda prompt (while `ccoligopred_env` is active):**

```bash
pip install ipykernel
python -m ipykernel install --user --name=ccoligopred_env --display-name "Python (CCOligoPred)"

```

When you open Jupyter Notebook, look at the top-right corner (or the Kernel menu) and select **`Python (CCOligoPred)`**.

---

## 🚀 3. Usage Guide

CCOligoPred can be executed directly from your terminal/command prompt or seamlessly within a Jupyter Notebook.

### Input Data Format

Ensure your input Excel file (`.xlsx`) contains these two columns:

* `Sequence`: The raw amino acid sequence.
* `Register`: The corresponding heptad registers annotation (e.g., abcdefg).
*(A sample file `Oligopred_trial_test.xlsx` is provided in the `datasets/` folder).*

### Method A: Running via Anaconda Prompt (Command Line)

Make sure your environment is activated (`conda activate ccoligopred_env`) and you are inside the `CCOligoPred` directory.

**To run the Full Prediction Pipeline (Multiclass + Binary):**

```bash
ccoligopred predict -i datasets/Oligopred_trial_test.xlsx -o ccoligopred_full_results.xlsx

```

**To extract Register-Based Features (RBF) only:**

```bash
ccoligopred rbf -i datasets/Oligopred_trial_test.xlsx -o standalone_rbf_matrix.xlsx

```

### Method B: Running via Jupyter Notebook

Due to how Windows and Jupyter manage terminal paths, it is highly recommended to explicitly set your working directory and use Python's module runner.

**Copy and paste this code block into your Jupyter cell to run the Full Prediction:**

```python
import os

# 1. Point Jupyter to your downloaded repository folder
# Change this path if you downloaded CCOligoPred to a different location
repo_path = r"C:\Users\YourUsername\CCOligoPred"  
os.chdir(repo_path)
print("Working Directory set to:", os.getcwd())

# 2. Run the Full Prediction!
!python -m ccoligopred.cli predict -i datasets/Oligopred_trial_test.xlsx -o ccoligopred_full_results.xlsx

```

**Copy and paste this code block to extract Features (RBF) only:**

```python
import os
os.chdir(r"C:\Users\YourUsername\CCOligoPred")

# Run RBF extraction
!python -m ccoligopred.cli rbf -i datasets/Oligopred_trial_test.xlsx -o standalone_rbf_matrix.xlsx

```

---

## 📁 Output Interpretations

Depending on the pipeline you choose to run, CCOligoPred will output the corresponding structured Excel file to your specified destination:

* **`ccoligopred predict` Output:** Generates an Excel file containing your original sequences alongside the multiclass predictions, binary predictions, and confidence probabilities.
* **`ccoligopred rbf` Output:** Bypasses the classifiers and strictly generates an Excel file containing your sequences mapped to their 2040 mathematical structural features.

```

```
