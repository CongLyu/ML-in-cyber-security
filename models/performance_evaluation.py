from sklearn.metrics import classification_report, accuracy_score
import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


def evaluate_and_plot_decision_tree(clf, X_test, y_test, feature_names,
                                    class_names, plot_path="tree_plot.png"):
    """
    Evaluate a Decision Tree Classifier and save the tree plot to a file.

    Parameters:
    - clf: trained classifier.
    - X_test: test features.
    - y_test: test target variable.
    - feature_names: names of the features for plotting.
    - class_names: names of the classes for plotting.
    - plot_path: path where to save the plot image.
    """
    # Predict the 'status' for the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier performance
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Plot the decision tree and save to file
    plt.figure(figsize=(20, 10))
    plot_tree(clf,
              filled=True,
              rounded=True,
              class_names=class_names,
              feature_names=feature_names,
              fontsize=10)

    # Save the plot to a file
    plt.savefig(plot_path)
    plt.close()  # Close the figure

    return plot_path
