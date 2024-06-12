# Introduction

The package provides two major functions:

- Train a model for predicting formation enthalpy using a customized dataset.
- Predict formation enthaly of new crystals with our pre-trained model.
 
##  Prerequisites

This package requires:

- python>=3.8
- scipy==1.7.1
- numpy
- pandas
- pymatgen>=2023.1.9
- xgboost
- scikit-learn

## Usage

Extracting source data from db file:
```bash
python3 extract.py
```
Note that this will generate a csv file and a directory containing the structure files. The csv file lists detailed information about the individual compounds, such as formation enthaly, data sources, etc.

Generating descriptors:
Before generating descriptors, first create a CSV file containing structures information by:
```bash
python3 utils.py --input="./testdata/poscar" --n_jobs=4 --output="structures.csv"
```
Then, generating descriptors by:
```bash
python3 descriptor.py --input="structures.csv" --n_jobs=4 --descriptor="All" --output="output.csv"
```
Note that the directory `./testdata/poscar` provided some poscar files, only for reference. You can carry out modification.

Train a model using the descriptors proposed in this work:
```bash
python3 model.py
```
Note that the directory `testdata` provided one csv file, only for reference. You can carry out modification.

Predict material formation enthalpy with our pre-trained model:
```bash
python3 predict.py < path_to_poscar_file >
```
