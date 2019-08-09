import multiprocessing
import os

import fasttext

import json
import csv

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

def GridSearch(dataFile, testFile):

    epoch = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    lr = (0.2, 0.4, 0.6, 0.8, 1.0)
    wordNgrams = (1, 2, 3, 4, 5)
    dim = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
    
    final=(0, 0, 0)
    with open("./.data/results.txt", "w") as file:
        for z in epoch:
            for y in lr:
                for x in wordNgrams:
                    for d in dim:
                        model = fasttext.train_supervised(
                                input=dataFile,
                                lr = y,
                                dim = d,
                                # ws=5,
                                epoch= z,
                                verbose = 2,
                                minCount = 1,
                                # minCountLabel=1,
                                # minn=0,
                                # maxn=0,
                                # neg=5,
                                wordNgrams= x,
                                loss="softmax",
                                # bucket=2000000,
                                thread = multiprocessing.cpu_count(),
                                # lrUpdateRate=100,
                                # t=0.0001
                                )
                        result = model.test("./.data/dev.txt")
                        print("result is {} and final is {} ".format(result, final))
                        if result > final:
                            print(result, 'is better than', final)
                            final = result
                            print("final is {}".format(final))
                            print("save model ...")
                            model.save_model("./.data/best_model.bin")
                            line = " with epoch: {}, lr: {}, wordNgrams:{}, \
                                     dim: {} \n".format(z, y, x, d)
                            file.write(final + line)

if __name__ == '__main__':
    recreate_model = False
    print_wrong_sentences = False
    test_model = True
    grid_search = True

    if recreate_model:
        recreate()
    
    # Create train file 
    train_data_source = '.data/snli/snli_1.0/snli_1.0_train_filtered.jsonl'
    train_labels_source = '.data/snli/snli_1.0/snli_1.0_train_gold_labels.csv'
    
    dataFile = "./.data/train.txt"
    
    processDataFile(dataFile, train_data_source, train_labels_source)
    
    # Create dev file
    dev_data_source = '.data/snli/snli_1.0/snli_1.0_dev_filtered.jsonl'
    dev_labels_source = '.data/snli/snli_1.0/snli_1.0_dev_gold_labels.csv'
    
    devFile = "./.data/dev.txt"

    processDataFile(devFile, dev_data_source, dev_labels_source)
    if grid_search:
        GridSearch(dataFile, devFile)

    model = fasttext.load_model("./.data/best_model.bin")
    
    if print_wrong_sentences:
        total = 0
        partial = {"neutral": 0, "contradiction": 0, "entailment": 0}
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
        result = model.test(devFile)
        print(model.test(devFile))
