# tools/utils/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """
    General confusion matrix plotting function
    """
    # 1. Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # 2. Print detailed report (view in console)
    print("\n" + "=" * 50)
    print("Detailed classification report (Classification Report):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("=" * 50 + "\n")

    # 3. Plot configuration
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 4. Save and display
    # Automatically create parent directory (if 'pictures' folder doesn't exist)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Confusion matrix heatmap saved to: {save_path}")

    # plt.show()
    plt.close()