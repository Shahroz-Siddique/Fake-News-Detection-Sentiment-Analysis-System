# src/models/train_transformer.py
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import mlflow
import mlflow.transformers
process_path= pd.read_csv('/home/foxtech/SHAHROZ_PROJ/Fake_news/data/processed/cleaned_news.csv')
# def load_data_for_transformer(processed_path):
#     df = pd.read_csv(processed_path)
#     return Dataset.from_pandas(df[['clean_content', 'label']])
#     print(dataset[0])


def load_data_for_transformer(processed_path):
    df = pd.read_csv(processed_path)
    df = df[['clean_content', 'label']]  # Ensure correct columns
    return Dataset.from_pandas(df)



# def tokenize_data(dataset):
#     tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#     return dataset.map(lambda x: tokenizer(x['clean_content'], padding='max_length', truncation=True), batched=True)
print(dataset)


def tokenize_data(dataset):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    return dataset.map(
        lambda batch: tokenizer(batch['clean_content'], padding='max_length', truncation=True),
        batched=True
    )



def train_distilbert_model(processed_path):
    dataset = load_data_for_transformer(processed_path)
    dataset = tokenize_data(dataset).train_test_split(test_size=0.2)
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        
        
    )
    tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    compute_metrics=compute_metrics



    with mlflow.start_run(run_name="DistilBERT"):
        trainer.train()
        mlflow.transformers.log_model(trainer.model, "distilbert_model")
