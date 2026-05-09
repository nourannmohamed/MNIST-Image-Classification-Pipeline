import numpy as np


def accuracy_score(y_true, y_pred):
    # convert labels to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # avoid division by zero
    if len(y_true) == 0:
        return 0.0

    # count correct predictions
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, labels=None):
    # convert labels to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # get all classes if labels are not given
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)

    # create empty matrix
    label_to_index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    # count true class vs predicted class
    for true_label, pred_label in zip(y_true, y_pred):
        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_label]
        matrix[true_index][pred_index] += 1

    return matrix


def precision_recall_f1(y_true, y_pred, labels=None, zero_division=0):
    # get confusion matrix first
    matrix = confusion_matrix(y_true, y_pred, labels)

    # get all classes if labels are not given
    if labels is None:
        labels = np.unique(np.concatenate((np.asarray(y_true), np.asarray(y_pred))))
    else:
        labels = np.asarray(labels)

    # store scores for each class
    scores = {}

    for i, label in enumerate(labels):
        # calculate values needed for metrics
        true_positive = matrix[i][i]
        false_positive = np.sum(matrix[:, i]) - true_positive
        false_negative = np.sum(matrix[i, :]) - true_positive
        support = np.sum(matrix[i, :])

        # calculate precision and recall
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        if precision_denominator == 0:
            precision = zero_division
        else:
            precision = true_positive / precision_denominator

        if recall_denominator == 0:
            recall = zero_division
        else:
            recall = true_positive / recall_denominator

        # calculate f1-score
        if precision + recall == 0:
            f1 = zero_division
        else:
            f1 = 2 * precision * recall / (precision + recall)

        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }

    return scores


def classification_report(y_true, y_pred, labels=None, zero_division=0):
    # convert labels to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # get all classes if labels are not given
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)

    # calculate all metric scores
    scores = precision_recall_f1(y_true, y_pred, labels, zero_division)
    accuracy = accuracy_score(y_true, y_pred)
    total_support = len(y_true)

    # create report header
    lines = []
    lines.append(f"{'':>12}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}")
    lines.append("")

    # add one row for each class
    for label in labels:
        label_scores = scores[label]
        lines.append(
            f"{str(label):>12}"
            f"{label_scores['precision']:>12.2f}"
            f"{label_scores['recall']:>12.2f}"
            f"{label_scores['f1-score']:>12.2f}"
            f"{label_scores['support']:>12}"
        )

    precisions = np.array([scores[label]["precision"] for label in labels])
    recalls = np.array([scores[label]["recall"] for label in labels])
    f1_scores = np.array([scores[label]["f1-score"] for label in labels])
    supports = np.array([scores[label]["support"] for label in labels])

    # calculate macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    # calculate weighted averages
    if total_support == 0:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
    else:
        weighted_precision = np.sum(precisions * supports) / total_support
        weighted_recall = np.sum(recalls * supports) / total_support
        weighted_f1 = np.sum(f1_scores * supports) / total_support

    # add final summary rows
    lines.append("")
    lines.append(f"{'accuracy':>36}{accuracy:>12.2f}{total_support:>12}")
    lines.append(
        f"{'macro avg':>12}"
        f"{macro_precision:>12.2f}"
        f"{macro_recall:>12.2f}"
        f"{macro_f1:>12.2f}"
        f"{total_support:>12}"
    )
    lines.append(
        f"{'weighted avg':>12}"
        f"{weighted_precision:>12.2f}"
        f"{weighted_recall:>12.2f}"
        f"{weighted_f1:>12.2f}"
        f"{total_support:>12}"
    )

    return "\n".join(lines)
