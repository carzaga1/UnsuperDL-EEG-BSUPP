import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

class ROCPlotter:
    def __init__(self, results_for_analysis):
        self.results_for_analysis = results_for_analysis
        self.auroc_scores = self.calculate_auroc_scores()

    def calculate_auroc_scores(self):
        auroc_scores = {}
        for patient_id, results in self.results_for_analysis.items():
            true_labels = results['labels']
            predictions = results['predictions']
            auroc = roc_auc_score(true_labels, predictions)
            auroc_scores[patient_id] = auroc
        return auroc_scores

    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))

        for patient_id, results in self.results_for_analysis.items():
            true_labels = results['labels']
            scores = results['predictions']

            fpr, tpr, _ = roc_curve(true_labels, scores)
            plt.plot(fpr, tpr, label=f'Patient {patient_id} (AUROC = {self.auroc_scores[patient_id]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal dashed line for reference
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Burst Suppression Detection')
        plt.legend(loc="lower right")
        plt.show()