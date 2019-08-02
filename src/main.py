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
                file.write(labels[read["annotator_labels"][0]] + " " + read["sentence1"] + "\n")


def recreate():
    datasets.SNLI.download(".data")

    dataFile = "./.data/data.txt"

    processDataFile(dataFile, '.data/snli/snli_1.0/snli_1.0_train.jsonl')

    # Skipgram model
    model = fasttext.train_supervised(dataFile)
    model.save_model("./.data/model_filename.bin")


if __name__ == '__main__':
    createPath = ".data/.created"
    if not os.path.isfile(createPath):
        open(createPath, "w")
        recreate()

    model = fasttext.load_model("./.data/model_filename.bin")

    correct = 0
    total = 0

    with jsonlines.open('.data/snli/snli_1.0/snli_1.0_test.jsonl') as reader:
        for obj in reader:
            read = reader.read(dict)

            correctLabel = labels[read["annotator_labels"][0]]

            (result, certanty) = model.predict(read["sentence1"])
            if result[0] == correctLabel:
                correct += 1

            total += 1

    print(correct/total)