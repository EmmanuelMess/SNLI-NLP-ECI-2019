#!/usr/bin/env python
import argparse
import json
import csv

def main():
    "Junta el archivo con las oraciones de test (jsonl)"
    " y los resultados de la clasificación de tu algoritmo (en tu formato)"
    " en un archivo csv compatible con el formato de Kaggle"

    sentences_filename = "snli_1.0/snli_1.0_test_filtered.jsonl"
    labels_filename = "test_cls.txt"
    output_filename = "result.csv"

    with open(output_filename, 'w') as fout:
        csv_writer = csv.writer(fout)
        csv_writer.writerow(['pairID', 'gold_label'])

        for pairID, label in it_ID_label_pairs(sentences_filename, labels_filename):
            formatted_label = format_label(label)
            csv_writer.writerow([pairID, formatted_label])

def format_label(label):
    return label[len("__label__"):]

def it_ID_label_pairs(sentences_filename, labels_filename):
    sentence_data = open(sentences_filename, 'r')
    labels_data = open(labels_filename, 'r')
    for pairID, label in zip(it_ID(sentence_data), it_labels(labels_data)):
        yield pairID, label

def it_ID(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield example['pairID']

def it_labels(label_data):
    for label in label_data:
        label = label.rstrip('\n')  # sacamos el fin de linea
        yield label




main()
