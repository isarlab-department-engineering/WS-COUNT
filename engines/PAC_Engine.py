# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
from engines.base_engine import base_engine

# import torchvision.models as models
# import torch

import torch.nn as nn

# from wildcat.olive import OliveCounting
# from wildcat.apple import AppleCounting
# from wildcat.almond import AlmondCounting
# from wildcat.car import CarCounting
# from wildcat.person import PersonCounting
# import random
# from scipy.ndimage import zoom

# import torchvision.transforms as transforms

# from tqdm import tqdm
# from tensorboardX import SummaryWriter
# import numpy as np
# from matplotlib import pyplot as plt

import os
import shutil
import time

# import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
from tqdm import tqdm

from util import AveragePrecisionMeter, Warp


class WC_Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 4

        if self._state('multi_gpu') is None:
            self.state['multi_gpu'] = False

        if self._state('device_ids') is None:
            self.state['device_ids'] = [0, 1, 2, 3]

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        # self.state['loss_batch'] = self.state['loss'].data[0]
        self.state['loss_batch'] = self.state['loss']
        self.state['meter_loss'].add(self.state['loss_batch'].item())

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        if not training:
            # input_var.volatile = True
            # target_var.volatile = True
            input_var.no_grad = True
            target_var.no_grad = True

        # compute output
        # self.state['output'] = model(input_var) # originale

        # ------ aggiunto! --------
        if self.state['use_gpu']:
            self.state['output'] = model(input_var.cuda())
        else:
            self.state['output'] = model(input_var)
        # -----------------------


        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            if self.state['multi_gpu']:
                model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            else:
                # model = torch.nn.DataParallel(model).cuda() # originale
                model = model.cuda()  # aggiunto!

            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            self.adjust_learning_rate(optimizer)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                # 'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(), # <------ originale
                'state_dict': model.state_dict(),  # <------ aggiunto
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']))

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)

        return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'],
                                             'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        if self.state['epoch'] is not 0 and self.state['epoch'] in self.state['epoch_step']:
            print('update learning rate')
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = param_group['lr'] * 0.1
                print(param_group['lr'])


class MAPEngine(WC_Engine):
    def __init__(self, state):
        WC_Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        WC_Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                # print(model.module.spatial_pooling)
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
            else:
                print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))

        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        WC_Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    # def obtain_weights(self, lbs):
    #     weights = []
    #     for i in range(0, lbs.shape[0]):
    #         if float(lbs[i]) < 0.5:
    #             # weights.append(1.0)
    #             # weights.append(20.0)
    #             weights.append(10.0)
    #
    #         else:
    #             # weights.append(10.0)
    #             weights.append(1.0)
    #     weights = [weights]
    #     weights = torch.from_numpy(np.array(weights, dtype=np.float32).T)
    #     weights = torch.autograd.Variable(weights.cuda())
    #     return weights

    def __on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])
        if self.state['use_gpu']:
            self.state['output'] = model(input_var.cuda())
        else:
            self.state['output'] = model(input_var)
        # weights_ = self.obtain_weights(target_var)
        # criterion.register_buffer('weight', weights_)
        self.state['loss'] = criterion(self.state['output'], target_var)
        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        if not training:
            with torch.no_grad():
                self.__on_forward(training, model, criterion, data_loader, optimizer, display)
        else:
            self.__on_forward(training, model, criterion, data_loader, optimizer, display)


class PAC_Engine(base_engine):

    def __init__(self, train_set=None, validation_set=None, test_set=None, model=None, device_id=0, lr=0.1, lrp=0.1,
                 save_path='',  momentum=0.9, weight_decay=1e-4, num_epochs=50,
                 image_size=-1, batch_size=24, workers=4, on_GPU=True):

        super(PAC_Engine, self).__init__()
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.model = model
        self.device_id = device_id
        self.lr = lr
        self.lrp = lrp
        self.save_path = save_path

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.workers = workers
        self.on_GPU = on_GPU

    def train_net(self):

        torch.cuda.set_device(self.device_id)

        # use_gpu = torch.cuda.is_available()
        # define dataset
        # train_dataset = OliveClassificationCnt(args.data, 'train2')
        # val_dataset = OliveClassificationCnt(args.data, 'test2-1')
        # num_classes = 1

        # load model
        # model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
        print('classifier', self.model.classifier)
        print('spatial pooling', self.model.spatial_pooling)

        # define loss function (criterion)
        criterion = nn.MultiLabelSoftMarginLoss()

        # define optimizer
        optimizer = torch.optim.SGD(self.model.get_config_optim(self.lr, self.lrp),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        state = {'batch_size': self.batch_size, 'image_size': self.image_size, 'max_epochs': self.num_epochs, 'evaluate': False, 'resume': None}
        state['difficult_examples'] = True
        state['save_model_path'] = self.save_path

        # engine = MultiLabelMAPEngine(state)
        engine = MAPEngine(state)
        engine.learning(self.model, criterion, self.train_set, self.validation_set, optimizer)

    def test_net(self):

        from models.PAC import PAC_Counter
        from util import init_dataset, extract_tiles, count_from_tiles, interval_rmse
        from torch.autograd import Variable
        import numpy as np
        from matplotlib import pyplot as plt

        model = PAC_Counter(self.model)
        model.eval()

        if self.on_GPU:
            model = model.cuda()

        val_dataset, val_loader = init_dataset(self.test_set, train=False,
                                               batch_size=self.batch_size, workers=self.workers)

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
            outputs = model.forward(inputs)  # .clamp(min=0)

            # first pyramid step
            tiles = extract_tiles(inputs)

            if self.on_GPU:
                tiles = Variable(tiles.cuda())
            else:
                tiles = Variable(tiles)

            out_tiles = model.forward(tiles)  # .clamp(min=0)
            out_tiles = count_from_tiles(out_tiles, labels.shape[0])

            # second pyramid step
            tiles2 = extract_tiles(tiles)

            if self.on_GPU:
                tiles2 = Variable(tiles2.cuda())
            else:
                tiles2 = Variable(tiles2)

            # print("16 tiles shape: ", tiles2.shape)
            out_tiles2 = model.forward(tiles2)  # .clamp(min=0)
            out_tiles2 = count_from_tiles(out_tiles2, labels.shape[0])

            # convert back to a numpy array
            if self.on_GPU:
                outputs = outputs.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                out_tiles = out_tiles.data.cpu().numpy()
                out_tiles2 = out_tiles2.data.cpu().numpy()
            else:
                outputs = outputs.data.numpy()
                labels = labels.data.numpy()
                out_tiles = out_tiles.data.numpy()
                out_tiles2 = out_tiles2.data.numpy()

            for b in range(outputs.shape[0]):
                pred = round(outputs[b, 0], 10)
                pred_t = round(out_tiles[b, 0], 10)
                pred_t2 = round(out_tiles2[b, 0], 10)

                lab = labels[b, 0]
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
        avg_pred = np.around((predictions + pred_tiles + pred_tiles2) / 3.0, 10)
        err_avg = labels_list - avg_pred

        print(val_errors.shape)

        # istogramma delle occorrenze degli errori
        plt.hist(val_errors, bins=100, color="skyblue", ec="black")  # arguments are passed to np.histogram
        plt.title("Errors Histogram (error = label - prediction)")
        plt.figure()
        plt.hist(predictions, bins=100, color="skyblue", ec="black")  # arguments are passed to np.histogram
        print("%s num samples: %f" % ('test', len(val_errors)))

        print("labels sum: ", labels_list.copy().sum())
        print("predictions sum: ", pred_tiles2.copy().sum())

        rmse = np.sqrt(np.square(val_errors).sum() / len(val_errors))
        rmse_tiles = np.sqrt(np.square(err_tiles).sum() / len(err_tiles))
        rmse_tiles2 = np.sqrt(np.square(err_tiles2).sum() / len(err_tiles2))
        rmse_avg = np.sqrt(np.square(err_avg).sum() / len(err_avg))

        print("root mean squared error: ", rmse)
        print("4tiles root mean squared error: ", rmse_tiles)
        print("16tiles root mean squared error: ", rmse_tiles2)
        print("avg root mean squared error: ", rmse_avg)

        plt.title(("Predictions Histogram - Root Mean Squared Error = %f" % (rmse)))
        plt.figure()
        # plt.stem(val_errors, label='error')  # arguments are passed to np.histogram
        plt.stem(labels_list, markerfmt='bo', label='GT')  # arguments are passed to np.histogram
        plt.stem(predictions, markerfmt='go', label='predictions')  # arguments are passed to np.histogram
        # plt.stem(pred_tiles2,'g', markerfmt='go', label='predictions')  # arguments are passed to np.histogram

        plt.legend()
        plt.title("GT vs Predictions")

        cond_rmse = interval_rmse(pred_tiles2, labels_list)

        plt.figure()
        plt.stem(cond_rmse, markerfmt='bo')  # arguments are passed to np.histogram
        plt.title("E2E conditioned rmse")
        plt.show()
