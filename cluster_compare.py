from sklearn import metrics
labels_true = [4, 4, 1, 1, 4, 4, 7]
labels_pred = [2, 2, 7, 7, 2, 2, 7]

print(metrics.normalized_mutual_info_score(labels_true, labels_pred) )
print(metrics.adjusted_mutual_info_score(labels_true, labels_pred) )
print(metrics.mutual_info_score(labels_true, labels_pred) )


labels_true = [4, 4, 1, 1, 4, 4, 7, 10, 6, 7]
labels_pred = [2, 2, 7, 7, 2, 2, 7, 10, 6, 7]

print(metrics.normalized_mutual_info_score(labels_true, labels_pred) )
print(metrics.adjusted_mutual_info_score(labels_true, labels_pred) )
print(metrics.mutual_info_score(labels_true, labels_pred) )