import torch

from torchtext import data
from torchtext import datasets

from  util import get_args

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
        print("With GPU")
    else:
        device = torch.device('cpu')
        print("With CPU")

    inputs = data.Field(lower=args.lower)
    answers = data.Field(sequential=False)

    train, validation, test = datasets.SNLI.splits(inputs, answers)

    

