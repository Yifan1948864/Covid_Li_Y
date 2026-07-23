FINAL REPORT: CODE EXECUTION AND OUTPUT GUIDE

1. Overview
------------------------------------------------------------

The "Code" folder contains all notebooks used for data cleaning,
data integration, model training, evaluation, feature-importance
estimation, and figure production.

Each .ipynb notebook can be opened and run independently, provided
that its required input files are already present. Each notebook
saves its results to the corresponding output folder. However, to
reproduce the complete analysis from the raw data, it is strongly
recommended that the notebooks be run in numerical order, beginning
with 01.


2. Important folder requirements
------------------------------------------------------------

The folder structure and folder names must remain exactly as provided:

Final Report/
|-- Code/
|-- Raw data/
|-- Cleaned data/
|-- Train and test/
|-- Model/
|-- Importance/
|-- Figure/
`-- README.txt

The notebooks use relative paths such as "../Raw data/" and
"../Cleaned data/". They should therefore be opened and run from the
"Code" folder. Do not rename or move the folders.

All output folders must already exist before the notebooks are run.
The code does not create missing output folders. If "Cleaned data",
"Train and test", "Model", "Importance", or "Figure" is absent, the
relevant notebook will be unable to save its results.


3. Software requirements
------------------------------------------------------------

Run the notebooks in a Python environment with Jupyter Notebook or
JupyterLab. The main required Python packages are:

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- optuna

Open the "Code" folder in Jupyter, open each notebook, and select
"Run All" to execute all cells in that notebook.

The file "07_model_functions.py" must remain in the "Code" folder.
It contains the shared modelling functions and is loaded automatically
by notebooks 08 and 09 using the %run command. It does not need to be
run separately.


4. Recommended execution order
------------------------------------------------------------

For a complete reproduction, run the .ipynb files in numerical order,
starting with 01 and ending with 10. Both 02_1 and 02_2 should be run
before continuing to 03. Each notebook can also be run independently
when all of its required input files and folders are already present.


5. Location and interpretation of files
------------------------------------------------------------

"Raw data" contains the originally downloaded source data and
supporting codebooks. These files are inputs and could not be replaced
by processed results.

Except for the source files in "Code", the original files in
"Raw data", and this README, the contents of the other folders are
results produced during execution:

- "Cleaned data": cleaned, merged, and policy-level datasets
- "Train and test": model training and held-out test datasets
- "Model": model parameters and evaluation results in JSON format
- "Importance": repeated feature-importance results in CSV format
- "Figure": figures used to report the analysis

Running the notebooks again may overwrite existing files with the same
names. Notebooks 08 and 09 perform model tuning and repeated fitting
and may take substantially longer to complete than the data-processing
notebooks.
