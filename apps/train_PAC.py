
from dataset.fruit_count_dataset import FruitCounting, FruitClassificationCnt
import models.PAC as pac
from engines.PAC_Engine import PAC_Engine
from configs import configs

if __name__ == "__main__":
    conf = configs()
    dataset_root = conf.dataset_root
    save_path = conf.PAC_model_path
    train_set = FruitClassificationCnt(root=dataset_root,
                                       set='train')
    validation_set = FruitClassificationCnt(root=dataset_root,
                                            set='test')
    test_set = FruitCounting(root=dataset_root,
                             set='test')
    save_path = save_path

    model = pac.resnet101_PAC(num_classes=1, pretrained=True, kmax=0.2, kmin=None, alpha=0.7, num_maps=8)

    engine = PAC_Engine(model=model, train_set=train_set, validation_set=test_set, test_set=test_set,
                        batch_size=23, save_path=save_path, num_epochs=50, lr=0.01, lrp=0.1)
    engine.train_net()
    engine.test_net()
