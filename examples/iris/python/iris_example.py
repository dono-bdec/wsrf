#!/usr/bin/env python3
"""
Iris Classification Example using Weighted Subspace Random Forest (WSRF)

This script demonstrates the usage of the wsrf package for classification
on the classic iris dataset. It mirrors the R implementation checks.

Run from the project root directory:
    python3.10 examples/iris/iris_example.py

Note: Ensure wsrf package is installed in your Python 3.10 environment:
    pip install -e .
"""

import sys
import os
import csv

import numpy as np

from wsrf import (
    WSRFClassifier,
    oob_error_rate,
    oob_confusion_matrix,
    strength,
    tree_correlation
)


class OutputRedirector:
    """Helper class to redirect output to both console and file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def main():
    # Set up output redirection
    output_file = os.path.join(os.path.dirname(__file__), 'iris_python_results.txt')
    redirector = OutputRedirector(output_file)
    sys.stdout = redirector
    
    try:
        print("=" * 70)
        print("WSRF Python Implementation - Iris Classification Example")
        print("=" * 70)
        print()
        
        # 1. Load the iris dataset
        print("Loading iris dataset...")
        data_path = os.path.join(os.path.dirname(__file__), 'iris.csv')
        
        # Read CSV file using built-in csv module
        X_list = []
        y_list = []
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                X_list.append([
                    float(row['sepal_length']),
                    float(row['sepal_width']),
                    float(row['petal_length']),
                    float(row['petal_width'])
                ])
                y_list.append(row['species'])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {sorted(set(y))}")
        print()
        
        # 2. Train/Test Split with seed for reproducibility (manual implementation)
        seed = 42
        print(f"Setting random seed to {seed} for reproducibility")
        
        # Set random seed
        np.random.seed(seed)
        
        # Create stratified train/test split manually
        # Get indices for each class
        unique_classes = np.unique(y)
        train_indices = []
        test_indices = []
        
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            np.random.shuffle(class_indices)
            
            # 70% train, 30% test
            split_point = int(0.7 * len(class_indices))
            train_indices.extend(class_indices[:split_point])
            test_indices.extend(class_indices[split_point:])
        
        # Convert to numpy arrays and shuffle
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print()
        
        # 3. Train the WSRF model
        print("Training the model...")
        model = WSRFClassifier(n_estimators=500, random_state=seed)
        model.fit(X_train, y_train)
        print()
        
        # 4. Display model summary
        print("=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)
        print(f"A Weighted Subspace Random Forest model with {model.n_estimators} trees.")
        print()
        # Calculate mtry using the same logic as the model
        from math import floor, log2
        mtry = max(1, floor(log2(X_train.shape[1])) + 1)
        print(f"  No. of variables tried at each split: {mtry}")
        print(f"        Minimum size of terminal nodes: {model.nodesize}")
        
        # 5. Calculate and display OOB statistics
        oob_error = oob_error_rate(model, X_train, y_train)
        print(f"                 Out-of-Bag Error Rate: {oob_error:.2f}")
        
        # Calculate strength and correlation
        model_strength = strength(model, X_train, y_train)
        
        # tree_correlation returns a correlation matrix, we need the mean off-diagonal correlation
        corr_matrix = tree_correlation(model, X_train)
        # Get mean of off-diagonal elements
        n_trees = corr_matrix.shape[0]
        if n_trees > 1:
            # Sum of all elements minus diagonal, divided by number of off-diagonal elements
            total_corr = np.sum(corr_matrix) - np.trace(corr_matrix)
            n_off_diag = n_trees * (n_trees - 1)
            model_correlation = total_corr / n_off_diag if n_off_diag > 0 else 0.0
        else:
            model_correlation = 0.0
        
        print(f"                              Strength: {model_strength:.2f}")
        print(f"                           Correlation: {model_correlation:.2f}")
        print()
        
        # 6. Display approximate confusion matrix
        # Note: This is an approximation using training set predictions
        # True OOB confusion matrix requires storing bootstrap masks
        print("Confusion matrix (training set approximation):")
        
        # Get predictions on training set
        train_predictions = model.predict(X_train)
        
        # Build confusion matrix
        class_labels = model.classes_
        n_classes = len(class_labels)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_label in enumerate(y_train):
            true_idx = np.where(class_labels == true_label)[0][0]
            pred_idx = np.where(class_labels == train_predictions[i])[0][0]
            confusion_matrix[true_idx, pred_idx] += 1
        
        # Calculate class errors
        class_errors = []
        for i in range(n_classes):
            total = confusion_matrix[i].sum()
            correct = confusion_matrix[i, i]
            error = (total - correct) / total if total > 0 else 0
            class_errors.append(error)
        
        # Print header
        max_label_len = max(len(str(label)) for label in class_labels)
        header = " " * (max_label_len + 1)
        for label in class_labels:
            header += f"{label:>11}"
        header += " class.error"
        print(header)
        
        # Print rows
        for i, label in enumerate(class_labels):
            row = f"{label:<{max_label_len}}"
            for j in range(n_classes):
                row += f"{confusion_matrix[i, j]:>11.0f}"
            row += f"        {class_errors[i]:.2f}"
            print(row)
        print()
        
        # 7. Test set predictions and accuracy
        print("=" * 70)
        print("TEST SET EVALUATION")
        print("=" * 70)
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Model Accuracy on Test Set: {accuracy * 100:.2f} %")
        print()
        
        print("=" * 70)
        print("Example completed successfully!")
        print(f"Results saved to: {output_file}")
        print("=" * 70)
        
    finally:
        # Restore stdout and close file
        sys.stdout = redirector.terminal
        redirector.close()
        print(f"\nResults have been saved to: {output_file}")


if __name__ == "__main__":
    main()
