from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ['Abortion', 'Economic Policy', 'Education Policy', 'Environmental Policy', 'Gay Rights', 'Military', 'Personal Responsibility',
                    'Social Views', 'Taxes', 'Worker\'s Rights']


def find_category(target_sequence):
    return classifier(target_sequence, candidate_labels, multi_label=True)
