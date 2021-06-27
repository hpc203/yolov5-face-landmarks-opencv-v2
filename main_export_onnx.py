import torch
import torch.nn as nn
import argparse
import cv2
import copy
import yaml
from models.experimental import attempt_load
from models.yolo import parse_model
from utils.datasets import letterbox
from utils.general import check_img_size
from models.common import Conv, Contract
from utils.activations import Hardswish, SiLU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
stride = [8, 16, 32]

def test_export(opt):
    ch = 3
    with open(opt.cfg) as f:
        yaml_info = yaml.load(f, Loader=yaml.FullLoader)

    anchors = yaml_info['anchors']
    nc = yaml_info['nc']
    na = len(anchors[0]) // 2
    no = nc + 5 + 10
    nl = len(anchors)

    _, save = parse_model(yaml_info, ch=[ch])
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    # Load model
    img_size = opt.imgsize
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(opt.image)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + opt.image
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    x = copy.deepcopy(img)
    onnxmodel = model.model
    y = []
    for m in onnxmodel:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)  # run
        y.append(x if m.i in save else None)  # save output
    print(torch.equal(x[0], pred))
    return onnxmodel, img, save, na, no

class my_yolov5_model(nn.Module):
    def __init__(self, model, save, na, no):
        super().__init__()
        self.model = model
        self.contract = Contract(gain=2)
        self.len_model = len(model)
        self.save = save
        self.na = na
        self.no = no
    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        x[0] = x[0].view(-1, self.no)
        x[1] = x[1].view(-1, self.no)
        x[2] = x[2].view(-1, self.no)
        return torch.cat(x, 0)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='yaml file path')
    parser.add_argument('--weights', type=str, default='weights/yolov5s-face.pt', help='model.pt path')
    parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imgsize', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()

    onnxmodel, img, save, na, no = test_export(opt)

    onnxmodel[-1].export = True
    net = my_yolov5_model(onnxmodel, save, na, no).to(device)
    net.eval()
    # with torch.no_grad():
    #     out = net(img)
    # print(out)

    f = opt.weights.replace('.pt', '.onnx')  # filename
    input = torch.zeros(1, 3, opt.imgsize, opt.imgsize).to(device)
    # Update model
    for k, m in net.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    torch.onnx.export(net, input, f, verbose=False, opset_version=12, input_names=['data'], output_names=['out'])

    cvnet = cv2.dnn.readNet(f)
    input = cv2.imread(opt.image)
    input = cv2.resize(input, (opt.imgsize,opt.imgsize))
    blob = cv2.dnn.blobFromImage(input)
    cvnet.setInput(blob)
    outs = cvnet.forward(cvnet.getUnconnectedOutLayersNames())