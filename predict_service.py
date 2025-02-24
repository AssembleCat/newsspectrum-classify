from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ['Abortion', 'Economic Policy', 'Education Policy', 'Environmental Policy', 'Gay Rights', 'Military', 'Personal Responsibility',
                    'Social Views', 'Taxes', 'Worker\'s Rights']


def find_category(target_sequence):
    results = []
    for single_sequence in target_sequence:
        result = classifier(single_sequence, candidate_labels, multi_label=True)
        results.append(result)
    return results

