import argparse
import dataloaders
import json
import models
import os
import torch
from train import get_instance


def main(config, path, device):
    if not os.path.exists(path):
        raise ValueError(f"File does not exist for ONNX conversion: {path}")

    train_loader = get_instance(dataloaders, 'train_loader', config)

    model = get_instance(models,
                         'arch',
                         config,
                         train_loader.dataset.num_classes)
    model.to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

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
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.model, args.device)
