import os

import numpy as np
import torch

from face_detection.alignment import load_net, batch_detect


def get_project_dir():
    current_path = os.path.abspath(os.path.join(__file__, "../"))
    return current_path


def relative(path):
    path = os.path.join(get_project_dir(), path)
    return os.path.abspath(path)


class RetinaFace:
    def __init__(
        self,
        gpu_id=-1,
        model_path=relative("weights/mobilenet0.25_Final.pth"),
        network="mobilenet",
    ):
        self.gpu_id = gpu_id
        self.device = (
            torch.device("cpu") if gpu_id == -1 else torch.device("cuda", gpu_id)
        )
        self.model = load_net(model_path, self.device, network)

    def detect(self, images):
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                return batch_detect(self.model, [images], self.device)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, self.device)
        elif isinstance(images, list):
            return batch_detect(self.model, np.array(images), self.device)
        elif isinstance(images, torch.Tensor):
            if len(images.shape) == 3:
                return batch_detect(self.model, images.unsqueeze(0), self.device)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, self.device)
        else:
            raise NotImplementedError()

    def __call__(self, images):
        return self.detect(images)

# import torch
# import torch.onnx as onnx
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model_path = relative("./weights/mobilenet0.25_Final.pth")  # 模型的路徑
# model = RetinaFace(gpu_id=0, model_path=model_path, network="mobilenet")  # 載入模型
# model.model.eval()  # 設置為評估模式

# dummy_input = torch.randn(1, 3, 640, 640).to(device)  # 示範輸入張量，根據您的模型需求進行調整
# onnx_file_path = "./face_detection/weights/retinaface.onnx"  # ONNX檔案的儲存路徑
# torch.onnx.export(model.model, dummy_input, onnx_file_path)

