
from dataset.fruit_count_dataset import FruitCounting
from models.SCOUNT import SCOUNT
from engines.SCOUNT_Engine import SCOUNT_Engine
from configs import configs

if __name__ == "__main__":
    conf = configs()
    dataset_root = conf.dataset_root
    save_path = conf.SCOUNT_model_path

    train_set = FruitCounting(root=dataset_root,
                              set='train')
    test_set = FruitCounting(root=dataset_root,
                             set='test')
    save_path = save_path
    log_path = save_path + '/log'

    # subsampled_dim1 and subsampled_dim2 are width_img/32 and height_img/32 approximate by excess
    model = SCOUNT(num_classes=1, num_maps=8, subsampled_dim1=conf.subsampled_dim1,
                   subsampled_dim2=conf.subsampled_dim2)
    engine = SCOUNT_Engine(model=model, train_set=train_set, validation_set=test_set, test_set=test_set, seed=1,
                           batch_size=23, save_path=save_path, log_path=log_path, num_epochs=50)
    results = engine.train_net()
    engine.test_net()
