# Swim Technique Error Detection

This project explores pose-based machine learning models for detecting freestyle swimming technique errors.

The goal is to compare simple baseline models with more advanced spatial-temporal models.

## Project Goal

Given a sequence of swimmer pose keypoints from freestyle video, the model predicts technique errors such as:

- dropped elbow
- crossing midline
- poor recovery
- timing-related stroke errors

## Models

This repository will include:

1. **Random Forest baseline**
   - Uses single-frame pose features.

3. **ST-GCN model**
   - Uses the swimmer skeleton as a graph.
   - Models both joint relationships and motion over time.

4. **ST-GCN with stroke phase prediction**
   - Predicts both stroke phase and technique error.
   - Uses stroke phases to aid accuracy in technique error prediction
   - This is the main proposed model.

## Current Status

The first goal is to build and test a Random Forest frame-by-frame baseline.

## Planned File Structure

```text
swim-technique-error-detection/
│
├── README.md
├── requirements.txt
├── models/
│   └── random_forest_baseline.py
├── train_lstm.py
└── data/
    └── README.md
