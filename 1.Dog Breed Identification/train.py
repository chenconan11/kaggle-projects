from kaggleImageFolder import KaggleImageFolder
from classifier import AngClassifier
from torchvision.transforms import transforms
import torch

from time import time
import argparse
import os

root = None
labels_csv = None
save_path = None

arch = None
hidden_units = None

use_gpu = None

split_p = None
batch_size = None
lr = None
epochs = None
save_per_iter = None


def read_data(data_dir, mode, split_p=0.2):
    if mode == 'train':
        data_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        data_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if mode == 'test':
        return KaggleImageFolder(data_dir, transform=data_transform, mode='test')
    else:
        return KaggleImageFolder(data_dir, labels_csv, mode=mode, transform=data_transform, split_p=split_p)


def accuracy(model, dataloader, n_data):
    correct = 0
    model.eval()
    for x, y in dataloader:
        z = model(x)
        yhat = torch.argmax(z, 1)
        correct += (y == yhat).sum().item()

    acc = correct / n_data
    return acc


def train(model, train_dataloader, valid_dataloader, n_train, n_val, criterion, optimizer):
    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    it = 1

    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            train_loss_list.append(loss.data.item())
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch: {}/{}...".format(epoch, epochs),
                      "Loss: {:.4f}".format(loss))

            if it % save_per_iter == 0:
                model.save_model(save_path+"/{}-{}-{}.pth".format(epoch, it, int(time())))
            it += 1

            if use_gpu:
                torch.cuda.empty_cache()

        train_acc_list.append(accuracy(model, train_dataloader, n_train))

        if valid_dataloader is not None:
            valid_acc_list.append(accuracy(model, valid_dataloader, n_val))
            print("Epoch: {}/{}... ".format(epoch + 1, epochs),
                  "Iter: {}".format(it),
                  "valid accuarcy: {:.4f}".format(valid_acc_list[-1]))
        else:
            print("Epoch: {}/{}... ".format(epoch + 1, epochs),
                  "Iter: {}".format(it),
                  "valid accuarcy: {:.4f}".format(train_acc_list[-1]))

        return train_loss_list, train_acc_list, valid_acc_list


def main():
    train_data = read_data(root + '/train', 'train', split_p=split_p)
    valid_data = read_data(root + '/train', 'valid', split_p=split_p)

    if valid_data is not None:
        assert train_data.n_classes == valid_data.n_classes, "train_data.n_classes != valid_data.n_classes"

    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size)

    n_class = train_data.n_classes
    model = AngClassifier(arch, hidden_units=hidden_units, n_class=n_class)

    optimizer = torch.optim.SGD(params=model.classifier.parameters(), momentum=0.9, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    n_train, n_val = len(train_data), len(valid_data)

    if use_gpu:
        model.cuda()
        criterion.cuda()
        torch.backends.cudnn.benchmark=True
        print('gpu for training...')
    else:
        print('cpu for training...')

    train_loss_list, train_acc_list, valid_acc_list = train(
        model, train_dataloader, valid_dataloader, n_train, n_val, criterion, optimizer)

    time_id = int(time())

    model.save_model(save_path+"/model-checkpoint-{}.pth".format(time_id))

    optimizer_params = {"optimizer_params": optimizer.state_dict(time_id)}
    torch.save(optimizer_params, save_path+"/optimizer-checkpoint-{}.pth".format(time_id))

    return train_loss_list, train_acc_list, valid_acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', help='path for data', default='F:/DATA/dog breed', type=str)
    parser.add_argument('--save_dir', help='save model path', default='./save_model', type=str)
    parser.add_argument('--labels_csv', help='labels.csv path', default='./labels.csv', type=str)
    parser.add_argument('--arch', help='chose pre-trained network(vgg13, vgg16)', default='vgg16', type=str)
    parser.add_argument('--split_p', help='split_p rate', default=0.2, type=float)
    parser.add_argument('--learning_rate', help='learning rate', default=0.01, type=float)
    parser.add_argument('--epochs', help='epochs', default=1, type=int)
    parser.add_argument('--save_per_iter', help='save model pre iterator', default=100, type=int)
    parser.add_argument('--batch_size', help='batch_size', default=16, type=int)
    parser.add_argument('--gpu', help='use gpu', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--hidden_units', help='hidden units for hidden layers', metavar='N', nargs='+',
                        default=[2048, 512], type=int)

    args = parser.parse_args()
    root = args.data_directory

    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    labels_csv = args.labels_csv
    arch = args.arch
    split_p = args.split_p
    use_gpu = args.gpu
    lr = args.learning_rate
    epochs = args.epochs
    save_per_iter = args.save_per_iter
    batch_size = args.batch_size
    hidden_units = args.hidden_units

    main()
