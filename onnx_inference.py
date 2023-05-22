# import argparse
# import os
# import numpy as np
# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from scipy import ndimage
# from tqdm import tqdm
# from math import ceil
# from glob import glob
# import onnxruntime as ort
# from PIL import Image
# import dataloaders
# import models
# from utils.helpers import colorize_mask
# from collections import OrderedDict

# def pad_image(img, target_size):
#     rows_to_pad = max(target_size[0] - img.shape[2], 0)
#     cols_to_pad = max(target_size[1] - img.shape[3], 0)
#     padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
#     return padded_img

# def sliding_predict(session, image, num_classes, flip=True):
#     image_size = image.shape
#     tile_size = (int(image_size[2]//2.5), int(image_size[3]//2.5))
#     overlap = 1/3

#     stride = ceil(tile_size[0] * (1 - overlap))
    
#     num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
#     num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
#     total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
#     count_predictions = np.zeros((image_size[2], image_size[3]))
#     tile_counter = 0

#     for row in range(num_rows):
#         for col in range(num_cols):
#             x_min, y_min = int(col * stride), int(row * stride)
#             x_max = min(x_min + tile_size[1], image_size[3])
#             y_max = min(y_min + tile_size[0], image_size[2])

#             img = image[:, :, y_min:y_max, x_min:x_max]
#             padded_img = pad_image(img, tile_size)
#             tile_counter += 1
#             ort_input = {session.get_inputs()[0].name: padded_img.cpu().numpy()}
#             padded_prediction = session.run(None, ort_input)[0]
#             if flip:
#                 flipped_ort_input = {session.get_inputs()[0].name: padded_img.flip(-1).cpu().numpy()}
#                 fliped_predictions = session.run(None, flipped_ort_input)[0]
#                 padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
#             predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
#             count_predictions[y_min:y_max, x_min:x_max] += 1
#             total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

#     total_predictions /= count_predictions
#     return total_predictions


# def multi_scale_predict(session, image, scales, num_classes, device, flip=False):
#     input_size = (image.size(2), image.size(3))
#     upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
#     total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

#     image = image.data.data.cpu().numpy()
#     for scale in scales:
#         scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
#         ort_input = {session.get_inputs()[0].name: scaled_img}
#         scaled_prediction = upsample(session.run(None, ort_input)[0])

#         if flip:
#             fliped_img = scaled_img.flip(-1)
#             ort_flipped_input = {session.get_inputs()[0].name: fliped_img.numpy()}
#             fliped_predictions = upsample(session.run(None, ort_flipped_input.cpu().numpy())[0])
#             scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
#         total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

#     total_predictions /= len(scales)
#     return total_predictions


# def save_images(image, mask, output_path, image_file, palette):
# 	# Saves the image, the model output and the results after the post processing
#     image_file = os.path.basename(image_file).split('.')[0]
#     colorized_mask = colorize_mask(mask, palette)
#     colorized_mask.save(os.path.join(output_path, image_file+'.png'))

# def main():
#     args = parse_arguments()
#     config = json.load(open(args.config))

#     # Dataset used for training the model
#     dataset_type = config['train_loader']['type']
#     assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K', 'DeepScene']
#     if dataset_type == 'CityScapes': 
#         scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] 
#     else:
#         scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
#     loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
#     to_tensor = transforms.ToTensor()
#     normalize = transforms.Normalize(loader.MEAN, loader.STD)
#     num_classes = loader.dataset.num_classes
#     palette = loader.dataset.palette

#     # Model
#     availble_gpus = list(range(torch.cuda.device_count()))
#     device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
#     session = ort.InferenceSession(args.model, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

#     if not os.path.exists(args.output):
#         os.makedirs(args.output)

#     image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
#     with torch.no_grad():
#         tbar = tqdm(image_files, ncols=100)
#         for img_file in tbar:
#             image = Image.open(img_file).convert('RGB')
#             input = normalize(to_tensor(image)).unsqueeze(0)
            
#             if args.mode == 'multiscale':
#                 prediction = multi_scale_predict(session, input, scales, num_classes, device)
#             elif args.mode == 'sliding':
#                 prediction = sliding_predict(session, input, num_classes)
#             else:
#                 ort_input = {session.get_inputs()[0].name: input.cpu().numpy()}
#                 prediction = session.run(None, ort_input)[0]
#                 prediction = prediction.squeeze(0)
#             prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
#             save_images(image, prediction, args.output, img_file, palette)

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Inference')
#     parser.add_argument('-c', '--config', default='VOC',type=str,
#                         help='The config used to train the model')
#     parser.add_argument('-mo', '--mode', default='multiscale', type=str,
#                         help='Mode used for prediction: either [multiscale, sliding]')
#     parser.add_argument('-m', '--model', default='model_weights.onnx', type=str,
#                         help='Path to the .onnx model checkpoint to be used in the prediction')
#     parser.add_argument('-i', '--images', default=None, type=str,
#                         help='Path to the images to be segmented')
#     parser.add_argument('-o', '--output', default='outputs', type=str,  
#                         help='Output Path')
#     parser.add_argument('-e', '--extension', default='jpg', type=str,
#                         help='The extension of the images to be segmented')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     main()



























from collections import OrderedDict
import dataloaders
import json
import models
import numpy as np
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from utils.helpers import colorize_mask
from utils import palette

config = json.load(open("config.json"))
IMAGE_PATH = "/home/markus/freiburg_forest_annotated/test/rgb/b1-09517_Clipped.jpg"
ONNX_MODEL = "/home/markus/pytorch-segmentation/saved/DeepSceneDeepLabV3+/05-13_23-34/best_model.onnx"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(MEAN, STD)
to_tensor = transforms.ToTensor()

img = Image.open(IMAGE_PATH).convert("RGB").crop((0, 0, 480, 480))
img.save("/home/markus/Pictures/crop_test.png")
# img = np.asarray(img, dtype=np.float32)
img = normalize(to_tensor(img)).unsqueeze(0).numpy()
print(img.shape)
# img = np.moveaxis(img, -1, 0)
# img = np.expand_dims(img, 0)

###############################################################################################

ort_session = onnxruntime.InferenceSession(ONNX_MODEL, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: img}
ort_outs = ort_session.run(None, ort_inputs)

print(type(ort_outs[0]))
print(ort_outs[0].shape)

prediction = ort_outs[0]
prediction = prediction.squeeze(0)
prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

colorized_mask = colorize_mask(prediction, palette.DeepScene_palette)
colorized_mask.save("/home/markus/Pictures/test.png")


################################################################################################
# Used to create the dummy input
loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
num_classes = loader.dataset.num_classes

# Model
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

# Load checkpoint
checkpoint = torch.load("/home/markus/pytorch-segmentation/saved/DeepSceneDeepLabV3+/05-13_23-34/best_model.pth", map_location=device)
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

img_tensor = torch.from_numpy(img).to(torch.device("cuda:0"))

prediction_pytorch = model(img_tensor)
prediction_pytorch = prediction_pytorch.squeeze(0)
prediction_pytorch = F.softmax(prediction_pytorch, dim=0).argmax(0).cpu().numpy()

colorized_mask = colorize_mask(prediction_pytorch, palette.DeepScene_palette)
colorized_mask.save("/home/markus/Pictures/test_torch.png")

np.testing.assert_allclose(prediction_pytorch, prediction, rtol=1e-03, atol=1e-05)
