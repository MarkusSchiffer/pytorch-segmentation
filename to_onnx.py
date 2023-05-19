import argparse
import os
import torch

def main(path: str, device: str, image_size: int) -> None:
    if not os.path.exists(path):
        raise ValueError(f"File does not exist for ONNX conversion: {path}")
    
    model = torch.load(path)
    dummy = torch.randn(1, image_size, image_size).to(device)
    file_name = os.path.splitext(path)[0] + ".onnx"
    torch.onnx.export(model,
                      dummy,
                      file_name,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"])

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='ONNX Model Conversion')
    parser.add_argument('-p', '--path', required=True, type=str,
                        help='Path to the PyTorch (.pth) file of the model to convert')
    parser.add_argument('-r', '--resolution', default=128, type=int,
                        help='size of one side of the input image, in pixes (default: 128)')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(args.path, args.device, args.resolution)