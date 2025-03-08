# mimic-iv-ecg-ve
Vector Embedding Pipeline for MIMIC-IV-ECG

This repository contains the Python files to create vector embeddings from 12-lead ECG data and to evaluate the effectiveness of vector embedding with downstream tasks.

### Step 1 
    Data handling and data curation for VE model training and evaluation with downstream tasks.

    001: EDA for MIMIC-IV-ECG 
    002: Diagnosis extraction from MIMIC-IV using ICD 
    003: Extraction of demographic information from MIMI-IV-hosp
    004: Data clustering for race
    005: Data selection for VE model training
    006: Creating TF dataset for faster training and evaluation
    007: Splitting data for training and evaluation

### Step 2
    011: VE model creation using VAE approach
    012: VE creation using trained VE model

### Step 3
    021: VE Evaluation with Downstream Tasks
