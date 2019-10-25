
from dataset.fruit_count_dataset import FruitCounting
from models.WSCOUNT import WSCOUNT
import models.PAC as pac
from engines.WSCOUNT_Engine import WSCOUNT_Engine
import torch
from configs import configs

if __name__ == "__main__":

    conf = configs()
    dataset_root = conf.dataset_root
    save_path = conf.WSCOUNT_model_path
    pac_load_path = conf.PAC_model_path
    train_set = FruitCounting(root=dataset_root,
                              set='train')
    test_set = FruitCounting(root=dataset_root,
                             set='test')
    save_path = save_path
    log_path = save_path + '/log'
    pac_load_path = pac_load_path

    # loading supervisor
    supervisor = pac.resnet101_PAC(num_classes=1, pretrained=True, kmax=0.2, kmin=None, alpha=0.7, num_maps=8)

    supervisor.load_state_dict(torch.load(pac_load_path)['state_dict'])

    # subsampled_dim1 and subsampled_dim2 are width_img/32 and height_img/32 approximate by excess
    model = WSCOUNT(num_classes=1, num_maps=8,
                    subsampled_dim1=conf.subsampled_dim1,
                    subsampled_dim2=conf.subsampled_dim2,
                    subsampled_t4_dim1=conf.subsampled_dim1_t4,
                    subsampled_t4_dim2=conf.subsampled_dim2_t4,
                    subsampled_t16_dim1=conf.subsampled_dim1_t16,
                    subsampled_t16_dim2=conf.subsampled_dim2_t16,
                    on_gpu=True, supervisor_model=supervisor)

    engine = WSCOUNT_Engine(model=model, train_set=train_set, validation_set=test_set, test_set=test_set, seed=1,
                            batch_size=11, save_path=save_path, log_path=log_path, num_epochs=50)
    results = engine.train_net()
    engine.test_net()
