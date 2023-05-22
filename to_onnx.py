import argparse
from collections import OrderedDict
import dataloaders
import json
import models
import os
import torch


def main(config, path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise ValueError(f"File does not exist for ONNX conversion: {path}")

    # Used to create the dummy input
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    num_classes = loader.dataset.num_classes

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # onnx does not support DataParallel, so remove module
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # create dummy rgb image as input
    image_size = config["train_loader"]["args"]["base_size"]
    dummy = torch.randn(1, 3, image_size, image_size).to(device)

    file_name = os.path.splitext(path)[0] + ".onnx"
    torch.onnx.export(model,
                      dummy,
                      file_name,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"])


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='ONNX Model Conversion')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file used for training \
                            (default: config.json)')
    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Path to the PyTorch (.pth) file of the model to \
                            convert')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    # if args.device:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.model)
