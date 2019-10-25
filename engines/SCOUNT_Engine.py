import torch
from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models

# from wildcat.olive import OliveCounting
# from wildcat.apple import AppleCounting
# from wildcat.almond import AlmondCounting
# from wildcat.car import CarCounting
# from wildcat.person import PersonCounting

# import torchvision.transforms as transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt
# from torch.nn.modules.batchnorm import BatchNorm1d
from util import conditioned_rmse, interval_rmse, init_dataset, set_seeds
from engines.base_engine import base_engine


class SCOUNT_Engine(base_engine):

    def __init__(self, model, train_set=None, validation_set=None, test_set=None, seed=123, batch_size=4,
                 workers=4, on_GPU=True, save_path='', log_path='', lr=0.0001, lrp=0.1,
                 momentum=0.9, weight_decay=1e-4, num_epochs=50):
        super(SCOUNT_Engine, self).__init__()

        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.seed = seed
        self.batch_size = batch_size
        self.workers = workers
        self.on_GPU = on_GPU
        self.save_path = save_path
        self.log_path = log_path
        self.lr = lr
        self.lrp = lrp
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

    def train_net(self
                  # train_set_,
                  #       val_set_,
                  #       writer,
                  #       seed_,
                  #       save_path
                  ):

        writer = SummaryWriter(self.log_path)
        best_validation_error = 1000.0
        set_seeds(self.seed)

        # dataset loaders
        train_dataset, train_loader = init_dataset(self.train_set, train=True, batch_size=self.batch_size, workers=self.workers)
        test_dataset, test_loader = init_dataset(self.validation_set, train=False, batch_size=self.batch_size, workers=self.workers)

        # load model
        self.model = self.model.train()
        if self.on_GPU:
            self.model = self.model.cuda()

        # define optimizer
        optimizer = torch.optim.SGD(self.model.get_config_optim(self.lr, self.lrp),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        criterion = nn.MSELoss()
        j = 0
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            train_loader = tqdm(train_loader, desc='Training')
            self.model.train()

            for i, data in enumerate(train_loader):
                # get the inputs
                inputs_datas, labels = data
                inputs, img_names = inputs_datas

                # wrap them in Variable
                if self.on_GPU:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                n_iter = j
                writer.add_scalar('Train/Loss', loss.data, n_iter)
                j += 1

            # save a new model for each epoch
            with torch.no_grad():
                validation_error = self.validate_current_model(val_loader=test_loader)
            print("validation error: ", validation_error)

            if validation_error < best_validation_error:
                best_validation_error = validation_error

                # save a new model for each epoch
                print('saving model: %s_epoch_%d' % (self.save_path, epoch))
                torch.save(self.model.state_dict(), ('%s/seed_%d_best_checkpoint.pth' % (self.save_path, self.seed)))

        print('Finished Training')
        return self.model, best_validation_error

    def validate_current_model(self, val_loader):
        self.model.eval()
        if self.on_GPU:
            self.model = self.model.cuda()

        val_errors = []
        val_loader = tqdm(val_loader, desc='Testing')
        predictions = []
        labels_list = []

        for i, data in enumerate(val_loader):
            # get the inputs
            inputs_datas, labels = data
            inputs, img_names = inputs_datas

            # wrap them in Variable
            if self.on_GPU:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = self.model.forward(inputs)

            # convert back to a numpy array
            if self.on_GPU:
                outputs = outputs.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
            else:
                outputs = outputs.data.numpy()
                labels = labels.data.numpy()

            for b in range(outputs.shape[0]):
                # lab = round(labels[b,0],10)
                lab = float(int(labels[b, 0]))

                pred = round(outputs[b, 0], 10)
                if pred < 0.0:
                    pred = 0.0
                err = lab - pred

                val_errors.append(err)
                predictions.append(pred)
                labels_list.append(lab)

        val_errors = np.array(val_errors)
        rmse = np.sqrt(np.square(val_errors).sum()/len(val_errors))
        return rmse

    def test_net(self):
        # define dataset
        test_dataset, test_loader = init_dataset(self.test_set, train=False, batch_size=self.batch_size, workers=self.workers)

        self.model.eval()

        if self.on_GPU:
            self.model = self.model.cuda()

        val_errors = []
        val_loader = tqdm(test_loader, desc='Testing')
        predictions = []
        labels_list = []
        for i, data in enumerate(val_loader):

            # get the inputs
            inputs_datas, labels = data
            inputs, img_names = inputs_datas

            # wrap them in Variable
            if self.on_GPU:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = self.model.forward(inputs)

            # convert back to a numpy array
            if self.on_GPU:
                outputs = outputs.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
            else:
                outputs = outputs.data.numpy()
                labels = labels.data.numpy()

            for b in range(outputs.shape[0]):
                lab = float(int(labels[b,0]))

                pred = round(outputs[b,0],10)
                if pred < 0.0:
                    pred = 0.0
                err = lab - pred

                val_errors.append(err)
                predictions.append(pred)
                labels_list.append(lab)

        val_errors = np.array(val_errors)
        predictions = np.array(predictions)
        labels_list = np.array(labels_list)

        print(val_errors.shape)

        # istogramma delle occorrenze degli errori
        plt.hist(val_errors, bins = 100, color = "skyblue", ec="black")  # arguments are passed to np.histogram
        plt.title("Errors Histogram (error = label - prediction)")
        plt.figure()
        plt.hist(predictions, bins = 100, color = "skyblue", ec="black")  # arguments are passed to np.histogram
        print("%s num samples: %f" % ('test' ,len(val_errors)))

        print("labels sum: ", labels_list.copy().sum())
        print("predictions sum: ", predictions.copy().sum())

        rmse = np.sqrt(np.square(val_errors).sum()/len(val_errors))

        print("root mean squared error: ", rmse)
        plt.title(("Predictions Histogram - Root Mean Squared Error = %f" % rmse))
        plt.figure()
        plt.stem(labels_list, markerfmt='bo', label='GT')  # arguments are passed to np.histogram
        plt.stem(predictions, markerfmt='go', label='predictions')  # arguments are passed to np.histogram
        plt.legend()
        plt.title("GT vs Predictions")

        cond_rmse = interval_rmse(predictions, labels_list)

        plt.figure()
        plt.stem(cond_rmse, markerfmt='bo')  # arguments are passed to np.histogram
        plt.title("E2E conditioned rmse")

        plt.show()

