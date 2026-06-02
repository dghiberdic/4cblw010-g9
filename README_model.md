# ML Module: FTIR Functional-Group Prediction

## Purpose

This module trains machine-learning models to predict functional-group presence or absence from FTIR spectra.

The project goal is to use these predictions as a screening tool for PET hydrolysis.

## Current model task

The task is multi-label classification.

This means one spectrum can contain multiple functional groups at the same time.

Example output:

- Ester: present
- Carboxylic acid: absent
- Alcohol/hydroxyl: present
- Arene: present

## Input data

The model expects:

```text
data/processed/X_spectra.npy
data/processed/y_labels.npy