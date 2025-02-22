from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ['Abortion', 'Economic Policy', 'Education Policy', 'Environmental Policy', 'Gay Rights', 'Military', 'Personal Responsibility',
                    'Social Views', 'Taxes', 'Worker\'s Rights']

target_sequence = "On the 20th, when the 10th hearing for President Yoon Seok-yeol's impeachment trial was held, supporters of President Yoon gathered and chanted, 'Invalidate the impeachment.' Pastor Jeon Kwang-hoon of Sarang Jeil Church made remarks at the rally, such as “Martial law must be imposed once again,” and “You must consult with me when running the government.”"

print(classifier(target_sequence, candidate_labels, multi_label=True))