import random
import pandas as pd
import json

file_path = './datasets/fine_tuned_doctor_dataset.csv'

df = pd.read_csv(file_path)

csv_data = []
for _, row in df.iterrows():
    diagnosis = row['label']
    symptoms = row['text']
    prompt = f"You are a medical diagnosis expert. You will give answer to patient's question based on the symptoms they have. Symptoms: '{symptoms}'. Question: 'What is the diagnosis I have?'. Response: You may be diagnosed with {diagnosis}."
    csv_data.append({"text": prompt})
random.shuffle(csv_data)
total_records = len(csv_data)
train_split = int(total_records * 2 / 3)
test_split = int(total_records * 1 / 6)
train_data = csv_data[:train_split]
test_data = csv_data[train_split:train_split + test_split]
valid_data = csv_data[train_split + test_split:]
with open('./fine_tuned_systems/train_data/fine_tuned_doctor/train.jsonl', 'w') as train_file:
    for entry in train_data:
        train_file.write(json.dumps(entry) + '\n')

with open('./fine_tuned_systems/train_data/fine_tuned_doctor/test.jsonl', 'w') as test_file:
    for entry in test_data:
        test_file.write(json.dumps(entry) + '\n')

with open('./fine_tuned_systems/train_data/fine_tuned_doctor/valid.jsonl', 'w') as valid_file:
    for entry in valid_data:
        valid_file.write(json.dumps(entry) + '\n')