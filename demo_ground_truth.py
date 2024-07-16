import argparse
import numpy as np
import cv2
import time
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from model.L2CS_Ghostnet import ResNet, Bottleneck, GhostModule

from face_detection.detector import RetinaFace
# from model.L2CS_Ghostnet import ResNet,Bottleneck,GhostModule
from model.L2CS import L2CS


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evaluation using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_1715138455/_epoch_27.pkl', type=str)
    parser.add_argument(
        '--image_dir', dest='image_dir', help='Directory of images to process.',
        default='image', type=str)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    batch_size = 1
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot
    image_dir = args.image_dir

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    model = ResNet(Bottleneck, [3, 4, 6, 3], 90, s=2, d=3)

    model2 = getArch(arch, 90)
    snapshot_path2 = "output/snapshots/L2CSNet_gaze360.pkl"

    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    saved_state_dict2 = torch.load(snapshot_path2)

    model.load_state_dict(saved_state_dict)
    model2.load_state_dict(saved_state_dict2)

    model.cuda(gpu)
    model2.cuda(gpu)
    model.eval()
    model2.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    with torch.no_grad():
        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            start_fps = time.time()
            faces = detector(frame)
            if faces is not None:
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min = int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = transformations(im_pil)
                    img = Variable(img).cuda(gpu)
                    img = img.unsqueeze(0)

                    # Gaze prediction
                    gaze_pitch, gaze_yaw = model(img)
                    gaze_pitch2, gaze_yaw2 = model2(img)

                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    pitch_predicted2 = softmax(gaze_pitch2)
                    yaw_predicted2 = softmax(gaze_yaw2)

                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    pitch_predicted2 = torch.sum(pitch_predicted2.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted2 = torch.sum(yaw_predicted2.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
                    pitch_predicted2 = pitch_predicted2.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted2 = yaw_predicted2.cpu().detach().numpy() * np.pi / 180.0

                    pp = [0.052776557192859876,-0.5430961488586578]
                    

                    # Draw gaze
                    draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (pp[0],pp[1]), color=(0, 255, 0))
                    #red ghost
                    draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (pitch_predicted, yaw_predicted), color=(0, 0, 255))
                    #yellow L2CS
                    draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (pitch_predicted2, yaw_predicted2), color=(0, 255, 255))

                    print(pp[0]-pitch_predicted,pp[1]-yaw_predicted)
                    print(pp[0]-pitch_predicted2,pp[1]-yaw_predicted2)

                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            output_path = os.path.join("output", img_name)
            cv2.imwrite(output_path, frame)
            cv2.imshow("Demo", frame)
            if cv2.waitKey(0) & 0xFF == 27:
                break

    # cv2.destroyAllWindows()
