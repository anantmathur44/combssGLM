#!/usr/bin/env python3
"""
Frank-Wolfe Demo for Best Subset Selection

This script demonstrates the Frank-Wolfe algorithm for logistic regression
best subset selection using the COMBSS framework.

"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import numpy as np
import logistic_fw as lfw
import metrics
import time


def main():
    """
    Main function to run the Frank-Wolfe demo for best subset selection.
    """
    
    # ========================================================================
    # DATA LOADING AND SETUP
    # ========================================================================
    
    print("Frank-Wolfe Demo for Best Subset Selection")
    print("=" * 50)
    
    # Load the CSV data
    #train_path = "./data/n-200-p1000Replica1.csv"
    #test_path = "./data/n-200-p1000Test-Replica1.csv"
    # Alternative smaller datasets for testing:
    train_path = "./data/Example_small_data_train.csv"
    test_path = "./data/Example_small_data_test.csv"

    print("Loading data...")
    data = np.loadtxt(train_path, delimiter='\t', skiprows=0)
    data_test = np.loadtxt(test_path, delimiter='\t', skiprows=0)

    # Split into X and y
    y_train = data[:, 0]  # First column is the response
    X_train = data[:, 1:]  # All remaining columns are features
    y_test = data_test[:, 0]  # First column is the response
    X_test = data_test[:, 1:]  # All remaining columns are features

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of test samples: {X_test.shape[0]}")

    # Note: X_train and X_test already include intercept column
    print(f"Training data already includes intercept: {X_train.shape}")
    print(f"Test data already includes intercept: {X_test.shape}")

    # Define true coefficients for performance evaluation
    # For demonstration purposes, assume features 1-10 are the true active features
    print(f"\n" + "="*60)
    print("TRUE MODEL SPECIFICATION (for performance evaluation)")
    print("="*60)
    p = X_train.shape[1]  # Total number of features
    beta_true = np.zeros(p)
    beta_true[:10] = 1.0  # First 10 features are active

    print(f"Total number of features: {p}")
    print(f"True active features: {np.where(beta_true != 0)[0] + 1}")  # +1 for 1-based indexing
    print(f"Number of true active features: {np.sum(beta_true != 0)}")
    print(f"Note: In real applications, you would use your actual true coefficients")

    # ========================================================================
    # FRANK-WOLFE MODEL FITTING
    # ========================================================================

    # Create and fit the Frank-Wolfe model
    print("\n" + "="*60)
    print("FRANK-WOLFE BEST SUBSET SELECTION DEMO")
    print("="*60)

    # Create model instance
    model = lfw.model()

    # Set parameters for Frank-Wolfe algorithm
    q = 10
    solver = 'sklearn'
    r = 1.5
    m = 10
    alpha = 0.05
    delta_min = 0.02
    cg_tol = 0.00001
    max_iter = 1000
    verbose = True  # Show progress as models are found

    print(f"\nRunning Frank-Wolfe algorithm with:")
    print(f"- Maximum subset size (q): {q}")
    print(f"- Solver: {solver}")
    print(f"- r parameter: {r}")
    print(f"- m parameter: {m}")
    print(f"- Alpha (line search): {alpha}")
    print(f"- Delta min: {delta_min}")
    print(f"- CG tolerance: {cg_tol}")
    print(f"- Maximum iterations: {max_iter}")
    print(f"- Verbose: {verbose}")
    print("\nFinding models for each subset size k...")

    # Fit the model using Frank-Wolfe algorithm
    start_time = time.time()
    model.fit(X_train, y_train,
              q=q, 
              solver=solver,
              r=r,
              m=m,
              alpha=alpha,
              delta_min=delta_min,
              cg_tol=cg_tol,
              max_iter=max_iter,
              verbose=verbose)
    end_time = time.time()

    print(f"\nFrank-Wolfe algorithm completed in {model.run_time:.2f} seconds")
    print(f"\nResults summary:")
    print(f"- Number of model sizes explored: {len(model.subset_list)}")
    print(f"- Total runtime: {model.run_time:.2f} seconds")

    # ========================================================================
    # BEST MODEL SELECTION AND EVALUATION
    # ========================================================================

    # Find the best model using metrics and evaluate performance
    print("\n" + "="*80)
    print("BEST MODEL SELECTION AND PERFORMANCE EVALUATION")
    print("="*80)

    # Calculate classification accuracy for each model
    print("\nEvaluating classification accuracy for each model size:")
    accuracies = []
    model_performances = []

    for i, model_features in enumerate(model.subset_list):
        if len(model_features) > 0:  # Only evaluate non-empty models
            accuracy = metrics.classification_acc(model_features, X_train, y_train, X_test, y_test)
            accuracies.append(accuracy)
            model_performances.append((i+1, model_features, accuracy))
            print(f"Model size {i+1}: Features {model_features}, Test Accuracy: {accuracy:.4f}")
        else:
            accuracies.append(0.0)
            model_performances.append((i+1, model_features, 0.0))
            print(f"Model size {i+1}: Empty model, Test Accuracy: 0.0000")

    # Find the best model (highest accuracy)
    best_idx = np.argmax(accuracies)
    best_accuracy = accuracies[best_idx]
    best_model_size = best_idx + 1
    best_features = model.subset_list[best_idx]
    best_iterations = model.niter_list[best_idx]

    print(f"\n" + "="*60)
    print("BEST MODEL FOUND")
    print("="*60)
    print(f"Best model size: {best_model_size}")
    print(f"Selected features: {best_features}")
    print(f"Feature indices (0-based): {[idx-1 for idx in best_features] if len(best_features) > 0 else []}")
    print(f"Test accuracy: {best_accuracy:.4f}")
    print(f"Iterations to converge: {best_iterations}")

    # ========================================================================
    # PERFORMANCE METRICS ANALYSIS
    # ========================================================================

    # Performance Metrics for Best Model
    print("\n" + "="*80)
    print("BEST MODEL PERFORMANCE METRICS")
    print("="*80)

    print("Using true coefficients defined earlier...")
    print(f"True active features: {np.where(beta_true != 0)[0] + 1}")  # +1 for 1-based indexing

    # Create predicted beta vector for best model
    beta_pred = np.zeros(p)
    if len(best_features) > 0:
        # Convert 1-based indices to 0-based for beta_pred
        selected_indices_0based = [idx - 1 for idx in best_features]
        beta_pred[selected_indices_0based] = 1.0

    # Compute performance metrics for best model
    perf_metrics = metrics.performance_metrics(beta_true, beta_pred)

    print(f"\n" + "="*60)
    print("BEST MODEL DETAILS")
    print("="*60)
    print(f"Model size: {best_model_size}")
    print(f"Selected features: {best_features}")
    print(f"Feature indices (0-based): {[idx-1 for idx in best_features]}")
    print(f"Iterations to converge: {best_iterations}")

    print(f"\n" + "="*60)
    print("CLASSIFICATION PERFORMANCE")
    print("="*60)
    print(f"Test accuracy: {best_accuracy:.4f}")

    print(f"\n" + "="*60)
    print("VARIABLE SELECTION PERFORMANCE")
    print("="*60)
    print(f"MCC (Matthews Correlation Coefficient): {perf_metrics['mcc']:.4f}")
    print(f"Variable selection accuracy: {perf_metrics['acc']:.4f}")
    print(f"Sensitivity (Recall): {perf_metrics['sens']:.4f}")
    print(f"Specificity: {perf_metrics['spec']:.4f}")
    print(f"F1 Score: {perf_metrics['f1']:.4f}")

    print(f"\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print(f"• MCC: Measures correlation between predicted and true feature selection")
    print(f"  (1.0 = perfect, 0.0 = random, -1.0 = completely wrong)")
    print(f"• Sensitivity: Proportion of true active features correctly identified")
    print(f"• Specificity: Proportion of true inactive features correctly identified")
    print(f"• F1 Score: Harmonic mean of precision and recall for feature selection")

    print(f"\nNote: Variable selection metrics compare predicted vs true feature selection.")
    print(f"In real applications, you would provide the actual true coefficients.")
    print(f"\nDemo completed successfully!")

    # Return results for potential further use
    return {
        'model': model,
        'best_features': best_features,
        'best_accuracy': best_accuracy,
        'performance_metrics': perf_metrics,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'beta_true': beta_true
    }


if __name__ == "__main__":
    # Run the demo when script is executed directly
    results = main()
