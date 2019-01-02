from classifier import AngClassifier
from train import read_data

import torch
import torch.nn.functional as F
import pandas as pd
import os

data_dir = 'f:/DATA/dog breed/test'
model_file = '''F:/DATA/dog breed/save_model/inception_v3-checkpoint-1546407881.pth'''
sample_submission = 'sample_submission.csv'


def predict():
    submission_samples = pd.read_csv(sample_submission)
    submission_samples = submission_samples.set_index('id')

    test_data = read_data(data_dir, 'test')
    test_images = test_data.samples

    data_len = len(test_data)
    print('test data len:', data_len)

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1)

    model = AngClassifier(load_file=model_file)
    model.eval()

    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()

    for i, (image, x) in enumerate(zip(test_images, test_loader)):
        print('{}/{}'.format(i, data_len))
        if cuda:
            x = x.cuda()
        z = model(x)
        prob = F.softmax(z, dim=1).cpu()[0].detach().numpy().tolist()
        image = os.path.basename(image).split('.')[0]

        submission_samples.loc[image] = prob

    submission_samples.to_csv('sample_submission_new.csv')


if __name__ == '__main__':
    predict()
