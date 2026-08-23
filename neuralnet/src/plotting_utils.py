import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def save_loss_curve(train_losses, val_losses, title, filename):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def save_accuracy_curve(train_acc, val_acc, title, filename):
    plt.figure()
    if train_acc is not None and len(train_acc) > 0:
        plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def save_confusion_matrix(conf_matrix, class_names, title, filename):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    fig, ax = plt.subplots()
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def save_f1_comparison_bar(class_names, f1_scores_dict, title, filename):
    x = np.arange(len(class_names))
    num_models = len(f1_scores_dict)
    width = 0.8 / num_models

    fig, ax = plt.subplots()
    
    for i, (model_name, f1_scores) in enumerate(f1_scores_dict.items()):
        offset = (i - num_models / 2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width, label=model_name)

    ax.set_ylabel('F1 Score')
    ax.set_title(title)
    ax.set_xticks(x, class_names)
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def save_feature_importance(feature_names, importance_values, title, filename):
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importance_values, align='center')
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()  # top to bottom
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
