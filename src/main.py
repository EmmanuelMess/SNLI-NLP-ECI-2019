import multiprocessing
import os

import fasttext

import json
import csv

from generate_answer import generate

labels = {"neutral": "__label__0", "contradiction": "__label__1",
          "entailment": "__label__2"}
inverse_labels = {v: k for k, v in labels.items()}


def it_sentences(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        text = example['sentence2']
        binary = example['sentence2_binary_parse']
        yield (text + binary) # sentence_2 and sentence2_binary_parse


def it_labels(label_data):
    label_data_reader = csv.DictReader(label_data)
    for example in label_data_reader:
        yield example['gold_label']


def processDataFile(resultPath, data_source, labels_source):
    with open(resultPath, "w") as file:
        sentence_data = open(data_source, 'r')
        label_data = open(labels_source, 'r')

        for sentence, label in zip(it_sentences(sentence_data),
                                   it_labels(label_data)):
            file.write(labels[label] + " " + sentence + "\n")


def recreate():
    train_data_source = '.data/snli/snli_1.0/snli_1.0_train_filtered.jsonl'
    train_labels_source = '.data/snli/snli_1.0/snli_1.0_train_gold_labels.csv'

    dataFile = "./.data/data.txt"

    processDataFile(dataFile, train_data_source, train_labels_source)

    model = fasttext.train_supervised(
        input=dataFile,
        lr=1.0,
        dim=325,
        # ws=5,
        epoch=25,
        verbose=2,
        minCount=1,
        # minCountLabel=1,
        # minn=0,
        # maxn=0,
        # neg=5,
        wordNgrams=2,
        loss="softmax",
        # bucket=2000000,
        thread=multiprocessing.cpu_count(),
        # lrUpdateRate=100,
        # t=0.0001
    )
    model.save_model("./.data/model_filename.bin")


if __name__ == '__main__':
    create_results = True

    if create_results:
        sentence_data = open(".data/snli/snli_1.0/snli_1.0_test_filtered.jsonl", 'r')
        output_labels = open(".data/test_cls.txt", 'w')
        model = fasttext.load_model("./.data/model_filename.bin")

        for sentence in it_sentences(sentence_data):
            output_labels.write("__label__" + inverse_labels[model.predict(sentence, k=1)[0][0]] + "\n")

        generate()
    else:
        recreate_model = True
        print_wrong_sentences = True
        test_model = True
        if recreate_model:
            recreate()

        dev_data_source = '.data/snli/snli_1.0/snli_1.0_dev_filtered.jsonl'
        dev_labels_source = '.data/snli/snli_1.0/snli_1.0_dev_gold_labels.csv'

        processDataFile("./.data/dev.txt", dev_data_source, dev_labels_source)

        model = fasttext.load_model("./.data/model_filename.bin")
        total = 0
        partial = {"neutral": 0, "contradiction": 0, "entailment": 0}
        if print_wrong_sentences:
            with open("./.data/dev.txt", "r") as file:
                for line in file:
                    correctLabel, sentence = line[:line.find(" ")],\
                        line[line.find(" "):].strip()

                    resultTuple = model.predict(sentence, k=1)
                    # print(resultTuple)
                    # print(correctLabel)
                    result = resultTuple[0][0]
                    resultConfidence = resultTuple[1][0]*100
                    if resultConfidence < 0:
                        result = "__label__0"
                    if result != correctLabel:
                        total += 1
                        partial[inverse_labels[correctLabel]] += 1
                        # print("'{}': chose {} with {}% confidence but was {}"
                        #       .format(sentence, inverse_labels[result],
                        #               resultConfidence,
                        #               inverse_labels[correctLabel]))

                print('Mistakes:', total)
                for k, v in labels.items():
                    print("{}: {} times".format(k, partial[k]))

        if test_model:
            print(model.test("./.data/dev.txt"))
