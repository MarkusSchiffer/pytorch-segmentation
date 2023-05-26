from collections import OrderedDict
import time
import json
import models
import numpy as np
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms
import torch
import onnx
import torch.nn.functional as F
from utils.helpers import colorize_mask
from utils import palette

config = json.load(open("config.json"))
IMAGE_PATH = "/root/catkin_ws/src/pytorch-segmentation/images/crop_test.png"
ONNX_MODEL = "/root/catkin_ws/src/best_model.onnx"
NUM_ITERATIONS = 100
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def to_numpy(tensor):
   return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

normalize = transforms.Normalize(MEAN, STD)
to_tensor = transforms.ToTensor()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

img = Image.open(IMAGE_PATH).convert("RGB").crop((0, 0, 480, 480))
img = normalize(to_tensor(img)).unsqueeze(0).to(device=device)
img = to_numpy(img)

###############################################################################################

"""
in order to work on Jetson, followed these steps:
pip3 install onnx
pip3 install onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl  # for jetpack 5.0
pip3 install tensorboard
"""

providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
# sess_options = onnxruntime.SessionOptions()
# sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
# sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
# ort_session = onnxruntime.InferenceSession(ONNX_MODEL, sess_options=sess_options, providers=providers)

onnx_model = onnx.load(ONNX_MODEL)
onnx.checker.check_model(onnx_model)

so = onnxruntime.SessionOptions()
so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

ort_session = onnxruntime.InferenceSession(ONNX_MODEL, so, providers=providers)

options = ort_session.get_provider_options()
cuda_options = options['CUDAExecutionProvider']
cuda_options['cudnn_conv_use_max_workspace'] = '1'
ort_session.set_providers(['CUDAExecutionProvider'], [cuda_options])

#IOBinding
input_names = ort_session.get_inputs()[0].name
output_names = ort_session.get_outputs()[0].name
io_binding = ort_session.io_binding()

io_binding.bind_cpu_input(input_names, img)
io_binding.bind_output(output_names, device.type)

#warm up run
ort_session.run_with_iobinding(io_binding)
ort_outs = io_binding.copy_outputs_to_cpu()

latency = []

for i in range(NUM_ITERATIONS):
    t0 = time.time()
    ort_session.run_with_iobinding(io_binding)
    latency.append(time.time() - t0)
    ort_outs = io_binding.copy_outputs_to_cpu()
print('Number of runs:', len(latency))
print("Average onnxruntime {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))


# ort_session = onnxruntime.InferenceSession(ONNX_MODEL, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: img}
# # run the first one, takes realllllly long
# ort_session.run(None, ort_inputs)
# print("ONNX starting")
# start = time.time()
# for i in range(NUM_ITERATIONS):
#     if i % 10 == 0:
#         print(f"onnx iteration: {i}")
#     ort_outs = ort_session.run(None, ort_inputs)
# duration = time.time() - start

# print(f"fps: {NUM_ITERATIONS/duration}")

# prediction = ort_outs[0]
# prediction = prediction.squeeze(0)
# prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

# colorized_mask = colorize_mask(prediction, palette.DeepScene_palette)
# colorized_mask.save("/home/markus/Pictures/test.png")


################################################################################################
# Used to create the dummy input
# loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
# num_classes = loader.dataset.num_classes
num_classes = 7

# Model
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

# Load checkpoint
checkpoint = torch.load("/root/catkin_ws/src/best_model.pth", map_location=device)
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

start = time.time()
for i in range(NUM_ITERATIONS):
    if i % 10 == 0:
        print(f"torch iteration: {i}")
    prediction_pytorch = model(img_tensor)
duration = time.time() - start
print(f"fps: {NUM_ITERATIONS/duration}")
# prediction_pytorch = prediction_pytorch.squeeze(0)
# prediction_pytorch = F.softmax(prediction_pytorch, dim=0).argmax(0).cpu().numpy()

# colorized_mask = colorize_mask(prediction_pytorch, palette.DeepScene_palette)
# colorized_mask.save("/home/markus/Pictures/test_torch.png")

# np.testing.assert_allclose(prediction_pytorch, prediction, rtol=1e-03, atol=1e-05)
