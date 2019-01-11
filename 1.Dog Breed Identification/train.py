
from classifier import AngClassifier, MixClassifier
from tools import accuracy, read_data
import torch
from torch.optim.lr_scheduler import StepLR

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


def train(model, train_dataloader, valid_dataloader, n_train, n_val, criterion, optimizer, epochs=20, scheduler=None):
    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    it = 1

    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step()

        epoch_losses = []
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            epoch_losses.append(loss.data.item())
            loss.backward()
            optimizer.step()

            it += 1

            if use_gpu:
                torch.cuda.empty_cache()

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}".format(mean_epoch_loss))
        train_loss_list.append(mean_epoch_loss)

        train_acc_list.append(accuracy(model, train_dataloader, n_train))

        if valid_dataloader is not None:
            acc = accuracy(model, valid_dataloader, n_val)
            if len(valid_acc_list) > 0 and acc > max(valid_acc_list):
                model.save_model(save_path + '/{}-checkpoint-best.pth'.format(model.arch))
                optimizer_params = {"optimizer_params": optimizer.state_dict()}
                torch.save(optimizer_params, save_path + "/{}-{}-checkpoint-best.pth".format(
                    model.arch, optimizer.__class__.__name__))
            valid_acc_list.append(acc)

            print("Epoch: {}/{}... ".format(epoch + 1, epochs),
                  "Iter: {}".format(it),
                  "valid accuracy: {:.4f}".format(valid_acc_list[-1]))
            print("Epoch: {}/{}... ".format(epoch + 1, epochs),
                  "Iter: {}".format(it),
                  "train accuracy: {:.4f}".format(train_acc_list[-1]))
        #
        # if epoch % save_per_iter == 0:
        #     model.save_model(save_path + '/{}-checkpoint-{}.pth'.format(model.arch, epoch))
        #     optimizer_params = {"optimizer_params": optimizer.state_dict()}
        #     torch.save(optimizer_params, save_path + "/{}-{}-checkpoint-{}.pth".format(
        #         model.arch, optimizer.__class__.__name__, epoch))

    return train_loss_list, train_acc_list, valid_acc_list


def main(model_file=None, param_file=None, use_mix=False):
    train_data = read_data(root + '/train', 'train', labels_csv, split_p=split_p)
    valid_data = read_data(root + '/train', 'valid', labels_csv, split_p=split_p)

    if valid_data is not None:
        assert train_data.n_classes == valid_data.n_classes, "train_data.n_classes != valid_data.n_classes"

    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    if len(valid_data) != 0:
        valid_dataloader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size)
    else:
        valid_dataloader = None

    if use_mix:
        if model_file is None:
            model = MixClassifier(n_classes=train_data.n_classes)
        else:
            model = MixClassifier(load_file=model_file)
    else:
        if model_file is None:
            n_class = train_data.n_classes
            model = AngClassifier(arch, hidden_units=hidden_units, n_classes=n_class)
        else:
            model = AngClassifier(load_file=model_file)

    # ignored_params = list(map(id, model.classifier.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params,
    #                      model.parameters())
    #
    # optimizer = torch.optim.Adam([
    #     {'params': base_params},
    #     {'params': model.classifier.parameters(), 'lr': lr}
    # ], lr=lr * 0.001)

    # optimizer = torch.optim.SGD(params=model.classifier.parameters(), momentum=0.9, lr=lr, weight_decay=1e-5)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    if use_gpu:
        model.cuda()
        criterion.cuda()
        torch.backends.cudnn.benchmark = True
        print('gpu for training...')
    else:
        print('cpu for training...')

    if param_file is not None:
        params = torch.load(param_file)
        optimizer.load_state_dict(params['optimizer_params'])

    n_train, n_val = len(train_data), len(valid_data)

    train_loss_list, train_acc_list, valid_acc_list = train(
        model, train_dataloader, valid_dataloader, n_train, n_val,
        criterion, optimizer, epochs=epochs, scheduler=scheduler)

    print('max valid accuracy: ', max(valid_acc_list))

    time_id = int(time())

    model.save_model(save_path+"/{}-checkpoint-{}.pth".format(model.arch, time_id))

    optimizer_params = {"optimizer_params": optimizer.state_dict()}
    torch.save(optimizer_params, save_path+"/{}-{}-checkpoint-{}.pth".format(
        model.arch, optimizer.__class__.__name__, time_id))

    return train_loss_list, train_acc_list, valid_acc_list


def test_train():
    global root
    global labels_csv
    global save_path

    global arch
    global hidden_units

    global use_gpu

    global split_p
    global batch_size
    global lr
    global epochs
    global save_per_iter

    root = 'F:/DATA/dog breed'

    save_path = os.path.join(root, 'save_model')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    labels_csv = './labels.csv'

    arches = ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
              'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
              'resnet18', 'resnet34', 'resnet50', 'resnet101',
              'resnet152', 'densenet121', 'densenet169', 'densenet201',
              'densenet161', 'squeezenet1_0', 'squeezenet1_1', 'inception_v3']

    split_p = 0.9
    use_gpu = True
    lr = 1e-2
    epochs = 1
    save_per_iter = 100
    batch_size = 16
    hidden_units = [2048, 1024]

    for test_arch in arches:
        arch = test_arch
        main()
    model_file = '''F:/DATA/dog breed/save_model/inception_v3-checkpoint-best.pth'''
    param_file = '''F:/DATA/dog breed/save_model/inception_v3-optimizer-checkpoint-best.pth'''
    main(model_file, param_file)


if __name__ == "__main__":
    # test_train()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', help='path for data', default='D:/DATA/dog breed', type=str)
    parser.add_argument('--save_dir', help='save model path', default='save_model', type=str)
    parser.add_argument('--labels_csv', help='labels.csv path', default='./labels.csv', type=str)
    parser.add_argument('--arch', help='chose pre-trained network(vgg13, vgg16)', default='resnet152', type=str)
    parser.add_argument('--split_p', help='split_p rate', default=0.2, type=float)
    parser.add_argument('--learning_rate', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--epochs', help='epochs', default=50, type=int)
    parser.add_argument('--save_per_iter', help='save model pre iterator', default=20, type=int)
    parser.add_argument('--batch_size', help='batch_size', default=8, type=int)
    parser.add_argument('--gpu', help='use gpu', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--hidden_units', help='hidadden units for hidden layers', metavar='N', nargs='+',
                        default=[1024, 512, 265], type=int)

    args = parser.parse_args()
    root = args.data_directory

    save_path = os.path.join(root, args.save_dir)
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

    main(use_mix=True)
    # model_file = '''D:/DATA/dog breed/save_model/MixClassifier-checkpoint-best.pth'''
    # param_file = '''D:/DATA/dog breed/save_model/MixClassifier-Adam-checkpoint-best.pth'''
    # main(model_file, None, use_mix=True)
