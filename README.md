# TDA-Histopathology

This repository contains code for the Topological Data Analysis used for Histpoathology Tumor image classification. 

Following files contains different code used for the experiments in this paper. All code files contains the information about each function implemented in is as comments.

## analysis.py
This file contains code for curves shown in the paper. Also it contains the code for generating different evaluation metrics using the confusion matrix of the test set results. 

## dataset_generation.py
This file contains code for generating tiles and all the features for the tiles. 

## feature_extraction.py
This file contains functions for extracting different features (Betti, LBP, Gabor)

## machine_learning
This file contains code for all machine learning related tasks (Simple training, 5 Fold CV, Train:Test Split, Random Forest, XGBoost, etc.)

## utils.py
This file contains utility functions written for the code (generating remaining time, OTSU thresholding)
