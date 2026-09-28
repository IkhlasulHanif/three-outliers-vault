import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import balanced_accuracy_score
import nlpaug.augmenter.word as naw
import ast

# Load the CSV file
file_path = 'train_eng_v2.csv'
df = pd.read_csv(file_path)

# Create a label mapping
label_mapping = {
    'Politik': 0,
    'Sosial Budaya': 1,
    'Ekonomi': 2,
    'Pertahanan dan Keamanan': 3,
    'Ideologi': 4,
    'Sumber Daya Alam': 5,
    'Demografi': 6,
    'Geografi': 7
}

# Apply the label mapping
df['label'] = df['label'].map(label_mapping)

# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

# Create an augmenter using BERT
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

# Function to augment data
def augment_data(df, target_count):
    augmented_texts = []
    augmented_labels = []
    
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        current_count = len(subset)
        
        while current_count < target_count:
            for text in subset['text']:
                augmented_text = aug.augment(text)[0]  # Take the string out of the list
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)
                current_count += 1
                if current_count >= target_count:
                    break

    augmented_df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
    return pd.concat([df, augmented_df], ignore_index=True)

# Augment the training data to ensure each class has at least 305 examples
train_df_augmented = augment_data(train_df, 305)
