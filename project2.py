import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression

class ModelSelector:
    def calculate_aic(self, model, X, y, model_type='linear'):
        """
        Calculate the AIC for a given model.
        """
        model.fit(X, y)

        n = len(y)

        # For regression (continuous target)
        if model_type in ['ridge', 'lasso', 'linear']:
            predictions = model.predict(X)
            residuals = y - predictions
            rss = sum(residuals**2)
            sigma_squared = rss / n
            log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_squared) + 1)
        # For classification (discrete target)
        elif model_type == 'logistic':
            probabilities = model.predict_proba(X)[:, 1]
            log_likelihood = 0
            for i in range(n):
                log_likelihood += y[i] * np.log(probabilities[i]) + (1 - y[i]) * np.log(1 - probabilities[i])
        else:
            raise ValueError("Unknown model type provided")

        k = len(model.coef_) + 1  # Number of coefficients (including the intercept)
        aic = 2 * k - 2 * log_likelihood
        return aic

    def k_fold_cross_validation(self, X, y, model, model_type='linear', k=5):
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // k
        aic_scores = []

        for i in range(k):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            aic_score = self.calculate_aic(model, X_train, y_train, model_type=model_type)
            aic_scores.append(aic_score)

        return aic_scores

    def bootstrap_validation(self, X, y, model, model_type='linear', n_bootstraps=100):
        n = len(y)
        aic_scores = []

        for _ in range(n_bootstraps):
            bootstrap_indices = np.random.choice(n, size=n, replace=True)
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]

            aic_score = self.calculate_aic(model, X_bootstrap, y_bootstrap, model_type=model_type)
            aic_scores.append(aic_score)

        return aic_scores

def summarize_results(aic_scores):
    n = len(aic_scores)
    mean_aic = sum(aic_scores) / n
    variance = sum((x - mean_aic) ** 2 for x in aic_scores) / n
    std_dev = variance ** 0.5
    return mean_aic, std_dev

def generic_process(data, X_columns, y_column):
    """
    Generic process to evaluate AIC for Linear, Ridge, and Lasso models on any dataset.
    """
    # Split features and target
    X = data[X_columns].values
    y = data[y_column].values

    # Initialize ModelSelector
    selector = ModelSelector()

    # Define models
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso()
    }

    results = {}

    for model_name, model in models.items():
        print(f"Evaluating model: {model_name.capitalize()}")

        # Perform k-Fold Cross-Validation
        aic_scores_kf = selector.k_fold_cross_validation(X, y, model, model_type=model_name, k=5)

        # Perform Bootstrapping
        aic_scores_bootstrap = selector.bootstrap_validation(X, y, model, model_type=model_name, n_bootstraps=100)

        # Calculate mean and standard deviation manually
        mean_aic_kf, std_aic_kf = summarize_results(aic_scores_kf)
        mean_aic_bootstrap, std_aic_bootstrap = summarize_results(aic_scores_bootstrap)

        # Save the results
        results[model_name] = {
            "mean_aic_kf": mean_aic_kf,
            "mean_aic_bootstrap": mean_aic_bootstrap,
            "kf_std_dev": std_aic_kf,
            "bootstrap_std_dev": std_aic_bootstrap,
            "kf_scores": aic_scores_kf,
            "bootstrap_scores": aic_scores_bootstrap
        }

    # Find the best model
    best_model = None
    best_mean_aic = float('inf')

    for model_name, scores in results.items():
        avg_aic = (scores["mean_aic_kf"] + scores["mean_aic_bootstrap"]) / 2
        print(f"\n{model_name.capitalize()} - Mean AIC:")
        print(f"  k-Fold: {scores['mean_aic_kf']:.3f}, Bootstrapping: {scores['mean_aic_bootstrap']:.3f}")
        print(f"  Std Dev (k-Fold): {scores['kf_std_dev']:.3f}, Std Dev (Bootstrap): {scores['bootstrap_std_dev']:.3f}")

        if avg_aic < best_mean_aic:
            best_mean_aic = avg_aic
            best_model = model_name

    print(f"\nBest Model: {best_model.capitalize()} with an average AIC of {best_mean_aic:.3f}")

    # Plotting the results
    plot_results(results)

def plot_results(results):
    """
    Plot the AIC results for k-Fold and Bootstrapping across models using horizontal bars.
    """
    models = list(results.keys())
    mean_aic_kf = [results[model]["mean_aic_kf"] for model in models]
    mean_aic_bootstrap = [results[model]["mean_aic_bootstrap"] for model in models]
    kf_std_dev = [results[model]["kf_std_dev"] for model in models]
    bootstrap_std_dev = [results[model]["bootstrap_std_dev"] for model in models]
    
    # Plot AIC Scores for k-Fold Cross-Validation
    fig, ax = plt.subplots(figsize=(8, 6))

    # k-Fold Horizontal Bar Plot with Error Bars
    ax.barh(models, mean_aic_kf, color='green', xerr=kf_std_dev, capsize=5)
    ax.set_title('k-Fold Cross-Validation AIC')
    ax.set_ylabel('Models')
    ax.set_xlabel('AIC')
    ax.set_yticks(range(len(models)))  # Set the ticks manually
    ax.set_yticklabels(models)  # Set the tick labels
    plt.show()  # Show k-Fold plot first

    # Bootstrap AIC Horizontal Bar Plot with Error Bars
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(models, mean_aic_bootstrap, color='purple', xerr=bootstrap_std_dev, capsize=5)
    ax.set_title('Bootstrap AIC')
    ax.set_ylabel('Models')
    ax.set_xlabel('AIC')
    ax.set_yticks(range(len(models)))  # Set the ticks manually
    ax.set_yticklabels(models)  # Set the tick labels
    plt.show()  # Show Bootstrap plot second

    # k-Fold Boxplot for Distribution of AIC Scores
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([results[model]["kf_scores"] for model in models], vert=True)
    ax.set_title('k-Fold AIC Distribution')
    ax.set_xticklabels(models)
    ax.set_ylabel('AIC')
    plt.show()  # Show k-Fold boxplot

    # Bootstrap Boxplot for Distribution of AIC Scores
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([results[model]["bootstrap_scores"] for model in models], vert=True)
    ax.set_title('Bootstrap AIC Distribution')
    ax.set_xticklabels(models)
    ax.set_ylabel('AIC')
    plt.show()  # Show Bootstrap boxplot


# Example Usage:
if _name_ == "_main_":
    # Load your dataset (replace 'your_dataset.csv' with the actual file path)
    data = pd.read_csv('others_dataset.csv')

    # Define feature columns and target column
    feature_columns = ['Age','Annual_Income','Spending_Score']  # Replace with your dataset's column names
    target_column = 'Store_Visits'  # Replace with your dataset's target column

    # Run the process
    generic_process(data, feature_columns, target_column)