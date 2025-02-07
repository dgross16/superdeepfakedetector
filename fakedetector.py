"""
eval pretained model.
"""
import os
import numpy as np
import yaml
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

from DeepfakeBench.training.metrics.utils import get_test_metrics
from DeepfakeBench.training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from DeepfakeBench.training.detectors import DETECTOR

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default=None,
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str,
                    default=None)
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DFDetector:
    def __init__(self, detector_config):

        print(detector_config.path)

        with open(detector_config.path, 'r') as f:
            config = yaml.safe_load(f)
        with open('./DeepfakeBench/training/config/test_config.yaml', 'r') as f:
            config2 = yaml.safe_load(f)

        config.update(config2)

        if 'label_dict' in config:
            config2['label_dict'] = config['label_dict']

        if config['pretrained']:
            config['pretrained'] = os.path.join("./DeepfakeBench", config['pretrained'])
        config['weights_path'] = config['pretrained']
        weights_path = config['weights_path']

        # prepare the model (detector)
        model_class = DETECTOR[config['model_name']]
        model = model_class(config).to(device)

        if weights_path:
            ckpt = torch.load(weights_path, map_location=device)
            model.load_state_dict(ckpt, strict=True)
            print('===> Load checkpoint done!')
        else:
            print('Fail to load the pre-trained weights')

        # set cudnn benchmark if needed
        if config['cudnn']:
            cudnn.benchmark = True

        # save config and model to DFDetector object
        self.config = config
        self.model = model
        self.test_data_loaders = None

    def get_config(self):
        return self.config

    def prepare_testing_data(self):
        def get_test_data_loader(config, test_name):
            # update the config dictionary with the specific testing dataset
            config = config.copy()  # create a copy of config to avoid altering the original one
            config['test_dataset'] = test_name  # specify the current test dataset
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
                )
            test_data_loader = \
                torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=config['test_batchSize'],
                    shuffle=False,
                    num_workers=int(config['workers']),
                    collate_fn=test_set.collate_fn,
                    drop_last=False
                )
            return test_data_loader

        test_data_loaders = {}

        for one_test_name in self.config['test_dataset']:
            test_data_loaders[one_test_name] = get_test_data_loader(self.config, one_test_name)

        self.test_data_loaders = test_data_loaders
        return


    def choose_metric(self):
        metric_scoring = self.config['metric_scoring']
        if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
            raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
        return metric_scoring


    def test_one_dataset(self, key):
        data_loader = self.test_data_loaders[key]
        model = self.model
        prediction_lists = []
        feature_lists = []
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            # get data
            data, label, mask, landmark = \
            data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
            label = torch.where(data_dict['label'] != 0, 1, 0)
            # move data to GPU
            data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
            if mask is not None:
                data_dict['mask'] = mask.to(device)
            if landmark is not None:
                data_dict['landmark'] = landmark.to(device)

            # model forward without considering gradient computation
            predictions = self.inference(model, data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())

        return np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)

    def test_epoch(self):
        model = self.model
        self.prepare_testing_data()
        test_data_loaders = self.test_data_loaders
        # set model to eval mode
        model.eval()

        # define test recorder
        metrics_all_datasets = {}

        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            data_dict = test_data_loaders[key].dataset.data_dict
            # compute loss for each dataset
            predictions_nps, label_nps, feat_nps = self.test_one_dataset(key)

            # compute metric for each dataset
            metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                                  img_names=data_dict['image'])
            metrics_all_datasets[key] = metric_one_dataset

            # info for each dataset
            for k, v in metric_one_dataset.items():
                if k == 'acc':
                    tqdm.write(f"{k}: {v}")

        return metrics_all_datasets

    @torch.no_grad()
    def inference(self, model, data_dict):
        predictions = model(data_dict, inference=True)
        return predictions

def main():
    # parse options and load config
    skip_please = ['iid.yaml', 'meso4.yaml', 'pcl_xception.yaml', 'altfreezing.yaml', 'sbi.yaml', 'lsda.yaml',
                   'xclip.yaml', 'videomae.yaml', 'i3d.yaml', 'sladd_detector.yaml', 'tall.yaml',
                   'sia.yaml', 'uia_vit.yaml', 'multi_attention.yaml', 'timesformer.yaml', 'lrl.yaml',
                   'sta.yaml', 'resnet34.yaml', 'f3net.yaml', 'ftcn.yaml', 'clip.yaml', 'spsl.yaml',
                   'facexray.yaml', 'ucf.yaml', 'efficientnetb4.yaml', 'recce.yaml']

    all_acc = []

    for detector_config in os.scandir('./DeepfakeBench/training/config/detector/'):
        if detector_config.name in skip_please:
            print(f"skipping {detector_config.name}")
            continue

        # Create DFDetector object
        detector = DFDetector(detector_config)
        config = detector.get_config()

        # start testing
        best_metric = detector.test_epoch()
        all_acc += [best_metric['temp']['acc']]

    print('===> Test Done!')
    return all_acc

if __name__ == '__main__':
    main()
