def calculate_f1_score(true_labels, predicted_labels):
    labels = set(true_labels + predicted_labels)
    f1_scores = []

    for label in labels:
        TP = sum(t == label and p == label for t, p in zip(true_labels, predicted_labels))
        FP = predicted_labels.count(label) - TP
        FN = true_labels.count(label) - TP

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0

def calculate_batch_f1_score(batch_true_labels, batch_predicted_labels):
    total_f1_score = 0
    num_examples = len(batch_true_labels)

    for true_labels, predicted_labels in zip(batch_true_labels, batch_predicted_labels):
        total_f1_score += calculate_f1_score(true_labels, predicted_labels)

    overall_f1_score = total_f1_score / num_examples
    return overall_f1_score


batch_true_labels = [["human", "human", "human", "fish"], ["dog", "cat"], ["bird", "bird", "fish"]]
batch_predicted_labels = [["human", "human", "fish"], ["dog", "dog"], ["bird", "fish", "fish"]]

overall_f1_score = calculate_batch_f1_score(batch_true_labels, batch_predicted_labels)
print("Overall F1 Score:", overall_f1_score)