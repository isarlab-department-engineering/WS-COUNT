import torch
from torch.autograd import Variable
# import torch.nn as nn
import torch.nn.functional as F
from engines.base_engine import base_engine
# import torchvision.models as models

# from wildcat.olive import OliveCounting
# from wildcat.apple import AppleCounting
# from wildcat.almond import AlmondCounting
# from wildcat.car import CarCounting
# from wildcat.person import PersonCounting
# import random
# from scipy.ndimage import zoom

# import torchvision.transforms as transforms

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt

# from torch.nn.modules.batchnorm import BatchNorm1d
from util import interval_rmse, init_dataset, \
     count_from_tiles, extract_loss_weights, class_from_tiles

# from scipy.misc import imresize
# from wildcat.models import resnet101_wildcat


class WSCOUNT_Engine(base_engine):
    def __init__(self, model, train_set=None, validation_set=None, test_set=None, seed=123, batch_size=4,
                 workers=4, on_GPU=True, save_path='', log_path='', lr=0.000001, lrp=0.1,
                 momentum=0.9, weight_decay=1e-4, num_epochs=50, sig_shift=0.3, sig_factor=10):
        super(WSCOUNT_Engine, self).__init__()

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
        self.sig_shift = sig_shift
        self.sig_factor = sig_factor

    def train_net(self):
        writer = SummaryWriter(self.log_path)

        best_validation_error = 1000.0

        # dataset loaders
        train_dataset, train_loader = init_dataset(self.train_set, train=True, batch_size=self.batch_size, workers=self.workers)
        test_dataset, test_loader = init_dataset(self.test_set, train=False, batch_size=self.batch_size, workers=self.workers)

        # load model
        if self.on_GPU:
            self.model = self.model.cuda()

        # define optimizer
        optimizer = torch.optim.SGD(self.model.get_config_optim(self.lr, self.lrp),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        j = 0
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            train_loader = tqdm(train_loader, desc='Training')
            # running_loss = 0.0
            # running_prediction = 0.0

            for i, data in enumerate(train_loader):
                # get the inputs
                inputs_datas, _ = data
                inputs, img_names = inputs_datas

                # wrap them in Variable
                if self.on_GPU:
                    inputs = Variable(inputs.cuda())
                else:
                    inputs = Variable(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                out_img, out_4tiles, out_16tiles = self.model.forward(inputs)
                labl_img, labl_4tiles, labl_16tiles = self.model.supervision(inputs)

                # ------ whole image scale
                out_img_class = F.sigmoid((out_img - self.sig_shift) * self.sig_factor)
                weights = extract_loss_weights(labl_img)
                loss_class_img = torch.nn.functional.binary_cross_entropy(out_img_class, labl_img, weight=weights)

                # print('(train) loss_class_img shape: ', loss_class_img.shape)

                # ----- first pyramid step
                count_4tiles = count_from_tiles(out_4tiles, inputs.shape[0])
                out_4tiles_class = F.sigmoid((out_4tiles - self.sig_shift)*self.sig_factor)
                weights_4t = extract_loss_weights(labl_4tiles)
                loss_class_4tiles = class_from_tiles(labl_4tiles, out_4tiles_class, weights_4t, inputs.shape[0]).mean()
                cross_loss_4t_img = F.mse_loss(count_4tiles, out_img)

                # ------ second pyramid step
                count_16tiles = count_from_tiles(out_16tiles, inputs.shape[0])
                out_16tiles_class = F.sigmoid((out_16tiles - self.sig_shift)*self.sig_factor)
                weights_16t = extract_loss_weights(labl_16tiles)
                loss_class_16tiles = class_from_tiles(labl_16tiles, out_16tiles_class, weights_16t, inputs.shape[0]).mean()
                cross_loss_16t_img = F.mse_loss(count_16tiles, out_img)
                cross_loss_16t_4t = F.mse_loss(count_16tiles, count_4tiles)

                # ---- total loss
                loss = loss_class_img + loss_class_4tiles + loss_class_16tiles + cross_loss_4t_img + cross_loss_16t_img + cross_loss_16t_4t

                loss.backward()
                optimizer.step()

                n_iter = j
                writer.add_scalar('Train/MSE_Loss', loss.data, n_iter)
                j += 1

            with torch.no_grad():
                validation_errors = self.validate_current_model(val_loader=test_loader)
            print("validation error: ", validation_errors)

            val_err, val_err_t4, val_err_t16, val_err_avg = validation_errors

            writer.add_scalar('Train/Validation_RMSE_1t', val_err, epoch)
            writer.add_scalar('Train/Validation_RMSE_4t', val_err_t4, epoch)
            writer.add_scalar('Train/Validation_RMSE_16t', val_err_t16, epoch)
            writer.add_scalar('Train/Validation_RMSE_AVG', val_err_avg, epoch)

            for error in [val_err, val_err_t4, val_err_t16, val_err_avg]:
                if error < best_validation_error:
                    best_validation_error = error

                    # save a new model for each epoch
                    print('saving model: %s_epoch_%d' % (self.save_path, epoch))
                    torch.save(self.model.state_dict(), ('%s/best_checkpoint.pth' % self.save_path))

            torch.save(self.model.state_dict(), ('%s/last_checkpoint.pth' % self.save_path))

        print('Finished Training')

    def validate_current_model(self, val_loader):
        self.model.eval()
        if self.on_GPU:
            model = self.model.cuda()

        val_errors = []
        val_loader = tqdm(val_loader, desc='Testing')
        err_tiles = []
        err_tiles2 = []

        predictions = []
        pred_tiles = []
        pred_tiles2 = []
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
            out_img, out_4tiles, out_16tiles = self.model.forward(inputs)
            # labl_img, labl_4tiles, labl_16tiles = self.model.supervision(inputs)

            # first pyramid step
            out_tiles = count_from_tiles(out_4tiles, labels.shape[0])
            # second pyramid step
            out_tiles2 = count_from_tiles(out_16tiles, labels.shape[0])

            # convert back to a numpy array
            if self.on_GPU:
                outputs = out_img.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                out_tiles = out_tiles.data.cpu().numpy()
                out_tiles2 = out_tiles2.data.cpu().numpy()
            else:
                outputs = out_img.data.numpy()
                labels = labels.data.numpy()
                out_tiles = out_tiles.data.numpy()
                out_tiles2 = out_tiles2.data.numpy()

            # outputs = np.array(outputs)#, dtype=np.int32)
            # labels = np.array(labels)#, dtype=np.int32)
            # errors = np.array(labels - outputs)#, dtype=np.int32)
            # print (outputs.shape)

            for b in range(outputs.shape[0]):

                pred = round(outputs[b, 0], 10)
                pred_t = round(out_tiles[b, 0], 10)
                pred_t2 = round(out_tiles2[b, 0], 10)

                lab = labels[b,0]
                err = lab - pred
                err_t = lab - pred_t
                err_t2 = lab - pred_t2

                val_errors.append(err)
                err_tiles.append(err_t)
                err_tiles2.append(err_t2)

                predictions.append(pred)
                pred_tiles.append(pred_t)
                pred_tiles2.append(pred_t2)

                labels_list.append(lab)

        val_errors = np.array(val_errors)
        predictions = np.array(predictions)
        pred_tiles = np.array(pred_tiles)
        labels_list = np.array(labels_list)
        pred_tiles2 = np.array(pred_tiles2)
        err_tiles2 = np.array(err_tiles2)
        err_tiles = np.array(err_tiles)
        avg_pred = np.around((predictions + pred_tiles + pred_tiles2)/3.0,10)
        err_avg = labels_list - avg_pred

        rmse = np.sqrt(np.square(val_errors).sum()/len(val_errors))
        rmse_tiles = np.sqrt(np.square(err_tiles).sum()/len(err_tiles))
        rmse_tiles2 = np.sqrt(np.square(err_tiles2).sum()/len(err_tiles2))
        rmse_avg = np.sqrt(np.square(err_avg).sum()/len(err_avg))

        return rmse, rmse_tiles, rmse_tiles2, rmse_avg

    def test_net(self):

        # dataset loaders
        val_dataset, val_loader = init_dataset(self.test_set, train=False, batch_size=self.batch_size, workers=self.workers)

        # load model
        self.model.eval()

        if self.on_GPU:
            model = self.model.cuda()

        val_errors = []
        val_loader = tqdm(val_loader, desc='Testing')
        err_tiles = []
        err_tiles2 = []

        predictions = []
        pred_tiles = []
        pred_tiles2 = []
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
            out_img, out_4tiles, out_16tiles = self.model.forward(inputs)
            # first pyramid step
            out_tiles = count_from_tiles(out_4tiles, labels.shape[0])
            # second pyramid step
            out_tiles2 = count_from_tiles(out_16tiles, labels.shape[0])

            # convert back to a numpy array
            if self.on_GPU:
                outputs = out_img.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                out_tiles = out_tiles.data.cpu().numpy()
                out_tiles2 = out_tiles2.data.cpu().numpy()
            else:
                outputs = out_img.data.numpy()
                labels = labels.data.numpy()
                out_tiles = out_tiles.data.numpy()
                out_tiles2 = out_tiles2.data.numpy()

            for b in range(outputs.shape[0]):

                pred = round(outputs[b, 0], 10)
                pred_t = round(out_tiles[b, 0], 10)
                pred_t2 = round(out_tiles2[b, 0], 10)

                lab = labels[b,0]
                err = lab - pred
                err_t = lab - pred_t
                err_t2 = lab - pred_t2

                val_errors.append(err)
                err_tiles.append(err_t)
                err_tiles2.append(err_t2)

                predictions.append(pred)
                pred_tiles.append(pred_t)
                pred_tiles2.append(pred_t2)

                labels_list.append(lab)

        val_errors = np.array(val_errors)
        predictions = np.array(predictions)
        pred_tiles = np.array(pred_tiles)
        labels_list = np.array(labels_list)
        pred_tiles2 = np.array(pred_tiles2)
        err_tiles2 = np.array(err_tiles2)
        err_tiles = np.array(err_tiles)
        avg_pred = np.around((predictions + pred_tiles + pred_tiles2)/3.0,10)
        err_avg = labels_list - avg_pred

        print(val_errors.shape)

        # istogramma delle occorrenze degli errori
        plt.hist(val_errors, bins = 100, color="skyblue", ec="black")  # arguments are passed to np.histogram
        plt.title("Errors Histogram (error = label - prediction)")
        plt.figure()
        plt.hist(predictions, bins = 100, color="skyblue", ec="black")  # arguments are passed to np.histogram
        print("%s num samples: %f" % ('test', len(val_errors)))

        print("labels sum: ", labels_list.copy().sum())
        print("predictions sum: ", pred_tiles2.copy().sum())

        rmse = np.sqrt(np.square(val_errors).sum()/len(val_errors))
        rmse_tiles = np.sqrt(np.square(err_tiles).sum()/len(err_tiles))
        rmse_tiles2 = np.sqrt(np.square(err_tiles2).sum()/len(err_tiles2))
        rmse_avg = np.sqrt(np.square(err_avg).sum()/len(err_avg))

        print("root mean squared error: ", rmse)
        print("4tiles root mean squared error: ", rmse_tiles)
        print("16tiles root mean squared error: ", rmse_tiles2)
        print("avg root mean squared error: ", rmse_avg)

        plt.title(("Predictions Histogram - Root Mean Squared Error = %f"% (rmse)))
        plt.figure()
        # plt.stem(val_errors, label='error')  # arguments are passed to np.histogram
        plt.stem(labels_list, 'b', markerfmt='bo', label='GT')  # arguments are passed to np.histogram
        # plt.stem(predictions,'g', markerfmt='go', label='predictions')  # arguments are passed to np.histogram
        plt.stem(pred_tiles2, 'g', markerfmt='go', label='predictions')  # arguments are passed to np.histogram

        plt.legend()
        plt.title("GT vs Predictions")
        cond_rmse = interval_rmse(pred_tiles, labels_list)

        plt.figure()
        plt.stem(cond_rmse, 'b', markerfmt='bo')  # arguments are passed to np.histogram
        plt.title("E2E conditioned rmse")
        plt.show()
