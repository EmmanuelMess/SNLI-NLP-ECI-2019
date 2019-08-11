import multiprocessing
import os

import fasttext

import json
import csv
import kaggle
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


def recreate(dataFile):
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

    epoch = (5)#, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    lr = (0.2)#, 0.4, 0.6, 0.8, 1.0)
    wordNgrams = (1, 2)#, 3, 4, 5)
    dim = (50)#, 100, 150, 200, 250, 300, 350, 400, 450, 500)
    
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
    download_data = True
    grid_search = True
    recreate_model = True
    print_wrong_sentences = False
    test_model = True
    create_results = True
    data_path = "./.data/"
    
    if download_data:
        for file in ('snli_1.0_train_filtered.jsonl',
            'snli_1.0_train_gold_labels.csv',
            'snli_1.0_dev_filtered.jsonl',
            'snli_1.0_dev_gold_labels.csv',
            'snli_1.0_test_filtered.jsonl'):
            download(data_path,file)   
            for item in os.listdir(data_path):
                if item.endswith('.zip'):
                    file_name = data_path+item
                    zip_ref = ZipFile(file_name) # create zipfile object
                    zip_ref.extractall(data_path) # extract file to dir
                    zip_ref.close() # close file
                    os.remove(file_name) # delete zipped file
                
    if grid_search or recreate_model:
      # Create train file 
      train_data_source = data_path + 'snli_1.0_train_filtered.jsonl'
      train_labels_source = data_path + 'snli_1.0_train_gold_labels.csv'

      trainFile = data_path + "train.txt"

      processDataFile(trainFile, train_data_source, train_labels_source)
    
    if print_wrong_sentences or test_model:
      # Create dev file
      dev_data_source = data_path + 'snli_1.0_dev_filtered.jsonl'
      dev_labels_source = data_path + 'snli_1.0_dev_gold_labels.csv'

      devFile = data_path + "dev.txt"

      processDataFile(devFile, dev_data_source, dev_labels_source)
    
    if grid_search:
        GridSearch(trainFile, devFile)
        model = fasttext.load_model("./.data/best_model.bin")
    
    if recreate_model:
        recreate(trainFile)

    if create_results:
        sentence_data = open(".data/snli_1.0_test_filtered.jsonl", 'r')
        model = fasttext.load_model("./.data/model_filename.bin")

        with open(".data/test_cls.txt", 'w') as output_labels:
            for sentence in it_sentences(sentence_data):
                output_labels.write("__label__" + inverse_labels[model.predict(sentence, k=1)[0][0]] + "\n")
                
        generate()
  
    model = fasttext.load_model("./.data/model_filename.bin")
        
    if print_wrong_sentences:
        sentence_data = open(dev_data_source, 'r')
        label_data = open(dev_labels_source, 'r')
        partial = {"neutral": 0, "contradiction": 0, "entailment": 0}
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
        print(model.test(devFile))
