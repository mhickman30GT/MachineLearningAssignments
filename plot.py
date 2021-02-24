from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(name, train_sizes, train_scores, test_scores, fit_times):
    """ Plot learning curve """
    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # Compute stats
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_title(f"Learning Curve for {name}")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Accuracy")
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")
    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    return fig


def plot_validation_curve(name, param_name, param_range, train_scores, test_scores):
    """ Plot validation curve """
    # Convert range to strings for non-singular params
    param_strings = [str(item) for item in param_range]

    # Generate figure
    fig, ax = plt.subplots(figsize=(8, 5))
    # Compute stats
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot validation curve
    plt.title(f"Validation Curves for {name} over range for {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    lw = 2
    plt.plot(
        param_strings, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_strings,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.plot(
        param_strings, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        param_strings,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    return fig


def plot_loss_curve(model):
    """ Plot the loss curve for NN """
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title('Neural Network Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(model.loss_curve_)
    plt.legend(loc="best")
    return fig


def plot_confusion_matrix(estimator, x_test, y_test):
    """ Plots confusion matrix """
    confusion_matrix_display = metrics.plot_confusion_matrix(
        estimator, x_test, y_test, cmap=plt.cm.Blues, normalize="true"
    )
    return confusion_matrix_display.ax_.get_figure()
