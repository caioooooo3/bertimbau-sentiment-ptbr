"""Inferência simples usando o modelo publicado no Hugging Face."""

import argparse
from transformers import pipeline

MODEL_ID = "caio16dev/bertimbau-sentiment-ptbr"


def main():
    parser = argparse.ArgumentParser(description="Classificação de sentimentos em português com BERTimbau.")
    parser.add_argument("text", type=str, help="Texto em português a ser classificado.")
    args = parser.parse_args()

    classifier = pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID)
    result = classifier(args.text)[0]

    print(f"Texto: {args.text}")
    print(f"Sentimento: {result['label']}")
    print(f"Confiança: {result['score']:.2%}")


if __name__ == "__main__":
    main()
