import torch

class ConfidenceThresholding:
    def __init__(self, confidence_threshold=0.7, uncertainty_threshold=0.1):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold

    def classify(self, confidences, uncertainties):
        labels = []
        for conf, unc in zip(confidences, uncertainties):
            if conf >= self.confidence_threshold and unc <= self.uncertainty_threshold:
                labels.append("Certain")
            elif conf >= self.confidence_threshold and unc > self.uncertainty_threshold:
                labels.append("Overconfident")
            else:
                labels.append("Uncertain")
        return labels

    def flag_predictions(self, confidences, uncertainties, true_labels, predictions):
        classification = self.classify(confidences, uncertainties)
        flagged = []
        for i, label in enumerate(classification):
            if label == "Overconfident" and predictions[i] != true_labels[i]:
                flagged.append(i)
        return flagged
