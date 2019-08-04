import multiprocessing
import os

import fasttext

import json
import csv

labels = {"neutral": "__label__0", "contradiction": "__label__1",
             "entailment": "__label__2"}


def it_sentences(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield example['sentence2']


def it_labels(label_data):
    label_data_reader = csv.DictReader(label_data)
    for example in label_data_reader:
        yield example['gold_label']


def processDataFile(resultPath, data_source, labels_source):
    with open(resultPath, "w") as file:
        sentence_data = open(data_source, 'r')
        label_data = open(labels_source, 'r')

        for sentence, label in zip(it_sentences(sentence_data), it_labels(label_data)):
            file.write(labels[label] + " " + sentence + "\n")


def recreate():
    data_source = '.data/snli/snli_1.0/snli_1.0_train_filtered.jsonl'
    labels_source = '.data/snli/snli_1.0/snli_1.0_train_gold_labels.csv'

    dataFile = "./.data/data.txt"

    processDataFile(dataFile, data_source, labels_source)

    model = fasttext.train_supervised(
        input=dataFile,
        #lr=1.0,
        dim=325,
        #ws=5,
        epoch=25,
        #verbose=2,
        minCount=1,
        #minCountLabel=1,
        #minn=0,
        #maxn=0,
        #neg=5,
        wordNgrams=2,
        #loss="softmax",
        #bucket=2000000,
        thread=multiprocessing.cpu_count(),
        #lrUpdateRate=100,
        #t=0.0001
    )
    model.save_model("./.data/model_filename.bin")


if __name__ == '__main__':
    createPath = ".data/.created"
    if not os.path.isfile(createPath):
        #open(createPath, "w")
        recreate()

    model = fasttext.load_model("./.data/model_filename.bin")

    data_source = '.data/snli/snli_1.0/snli_1.0_dev_filtered.jsonl'
    labels_source = '.data/snli/snli_1.0/snli_1.0_dev_gold_labels.csv'

    processDataFile("./.data/dev.txt", data_source, labels_source)

    print(model.test("./.data/dev.txt"))