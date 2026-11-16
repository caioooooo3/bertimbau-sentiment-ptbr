"""Pipeline reproduzível de fine-tuning do BERTimbau para análise de sentimentos em PT-BR."""

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
PARQUET_URL = "https://huggingface.co/datasets/ruanchaves/b2w-reviews01/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
MAX_LENGTH = 128
RANDOM_STATE = 42


def load_and_prepare_data():
    dataset = load_dataset("parquet", data_files=PARQUET_URL, split="train")
    df = dataset.to_pandas()

    df_model = df[df["overall_rating"].isin([1, 2, 4, 5])].copy()
    df_model = df_model.dropna(subset=["review_text"])
    df_model = df_model[df_model["review_text"].str.strip() != ""]
    df_model["label"] = (df_model["overall_rating"] >= 4).astype(int)

    label_count_per_text = df_model.groupby("review_text")["label"].nunique()
    conflicting_texts = label_count_per_text[label_count_per_text > 1].index

    df_clean = df_model[~df_model["review_text"].isin(conflicting_texts)].copy()
    df_clean = df_clean.drop_duplicates(subset=["review_text"])
    df_clean = df_clean[["review_text", "label"]].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df_clean,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df_clean["label"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    return train_df, valid_df, test_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
    }


def main():
    train_df, valid_df, test_df = load_and_prepare_data()

    print(f"Treino: {train_df.shape}")
    print(f"Validação: {valid_df.shape}")
    print(f"Teste: {test_df.shape}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    def tokenize_function(batch):
        return tokenizer(batch["review_text"], truncation=True, max_length=MAX_LENGTH)

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    valid_tokenized = valid_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    id2label = {0: "NEGATIVO", 1: "POSITIVO"}
    label2id = {"NEGATIVO": 0, "POSITIVO": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="./bertimbau-sentiment-ptbr",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=200,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nValidação:")
    print(trainer.evaluate())

    print("\nTeste:")
    print(trainer.predict(test_tokenized).metrics)

    trainer.save_model("./bertimbau-sentiment-ptbr-final")
    tokenizer.save_pretrained("./bertimbau-sentiment-ptbr-final")


if __name__ == "__main__":
    main()
