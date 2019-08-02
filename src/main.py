import multiprocessing
import os

import fasttext
import jsonlines

from torchtext import datasets

labels = {"neutral": "__label__0", "contradiction": "__label__1", "entailment": "__label__2"}


def processDataFile(resultPath, sourcePath):
    with open(resultPath, "w") as file:
        with jsonlines.open(sourcePath) as reader:
            for obj in reader:
                read = reader.read(dict)
                file.write(labels[read["annotator_labels"][0]] + " " + read["sentence2"] + "\n")


def recreate():
    datasets.SNLI.download(".data")

    dataFile = "./.data/data.txt"

    processDataFile(dataFile, '.data/snli/snli_1.0/snli_1.0_train.jsonl')

    model = fasttext.train_supervised(
        dataFile,
        #lr=1,
        dim=325,
        #ws=5,
        epoch=10,
        #minCount=1,
        #minCountLabel=1,
        #minn=0,
        #maxn=0,
        #neg=5,
        #wordNgrams=1,
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

    processDataFile("./.data/test.txt", '.data/snli/snli_1.0/snli_1.0_test.jsonl')

    print(model.test("./.data/test.txt"))