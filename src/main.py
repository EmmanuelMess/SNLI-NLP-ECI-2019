import multiprocessing
import os

import fasttext

import json
import csv
from zipfile import ZipFile

from generate_answer import generate

labels = {"neutral": "__label__0", "contradiction": "__label__1",
          "entailment": "__label__2"}
inverse_labels = {v: k for k, v in labels.items()}

def download(data_path,file):
    kaggle.api.competition_download_file('eci2019nlp',file, path=data_path)

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
    train_data_source = '.data/snli_1.0_train_filtered.jsonl'
    train_labels_source = '.data/snli_1.0_train_gold_labels.csv'

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
    download_data = False
    create_results = True
    recreate_model = True

    data_path = "./.data/"

    dev_data_source = data_path + 'snli_1.0_dev_filtered.jsonl'
    dev_labels_source = data_path + 'snli_1.0_dev_gold_labels.csv'

    if download_data:
        import kaggle

        for file in ('snli_1.0_train_filtered.jsonl',
             'snli_1.0_train_gold_labels.csv',
             'snli_1.0_dev_filtered.jsonl',
             'snli_1.0_dev_gold_labels.csv',
             'snli_1.0_test_filtered.jsonl'):
            download(data_path,file)   
            for item in os.listdir(data_path):
                print(item)
                if item.endswith('.zip'):
                    file_name = data_path+item
                    zip_ref = ZipFile(file_name) # create zipfile object
                    zip_ref.extractall(data_path) # extract file to dir
                    zip_ref.close() # close file
                    os.remove(file_name) # delete zipped file

    if recreate_model:
        recreate()

    if create_results:
        sentence_data = open(".data/snli_1.0_test_filtered.jsonl", 'r')
        model = fasttext.load_model("./.data/model_filename.bin")

        with open(".data/test_cls.txt", 'w') as output_labels:
            for sentence in it_sentences(sentence_data):
                output_labels.write("__label__" + inverse_labels[model.predict(sentence, k=1)[0][0]] + "\n")

        generate()
    else:
        print_wrong_sentences = True
        test_model = True

        model = fasttext.load_model("./.data/model_filename.bin")
        partial = {"neutral": 0, "contradiction": 0, "entailment": 0}
        if print_wrong_sentences:
            sentence_data = open(dev_data_source, 'r')
            label_data = open(dev_labels_source, 'r')

            total = 0
            totalMistakes = 0

            for sentence, correctLabelRaw in zip(it_sentences(sentence_data), it_labels(label_data)):
                correctLabel = labels[correctLabelRaw]
                resultTuple = model.predict(sentence, k=1)
                result = resultTuple[0][0]
                resultConfidence = resultTuple[1][0]*100
                total += 1
                if result != correctLabel:
                    totalMistakes += 1
                    partial[inverse_labels[correctLabel]] += 1
                    # print("'{}': chose {} with {}% confidence but was {}"
                    #       .format(sentence, inverse_labels[result],
                    #               resultConfidence,
                    #               inverse_labels[correctLabel]))

            print("Total: ", total)
            print('Mistakes:', totalMistakes)
            for k, v in labels.items():
                print("{}: {} times".format(k, partial[k]))

        if test_model:
            processDataFile("./.data/dev.txt", dev_data_source, dev_labels_source)
            print(model.test("./.data/dev.txt"))
