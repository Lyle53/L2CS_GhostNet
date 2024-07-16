import os
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import matplotlib.pyplot as plt
import datasets
# from model.L2CS_MobileOne import L2CS
from model.L2CS_Ghostnet import ResNet,Bottleneck,GhostModule

from utils import select_device
from torch.optim.lr_scheduler import ReduceLROnPlateau

##### 參數設定 #####
#資料集位置
gaze360image_dir = './datasets/gaze360_RGB/Image'
gaze360label_dir = './datasets/gaze360_RGB/Label/train.label'
gaze360val_label_dir = './datasets/gaze360_RGB/Label/val.label'
output           = './output/snapshots/'




gpu_id = "0"
batch_size = 2
gpu = select_device(gpu_id, batch_size=batch_size)
num_epochs = 60
arch = 'L2CS_GhostNet'
alpha = 1.0

lr = 0.00001
data_set = "Gaze360"
##trainsformation
transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


#要從頭訓練L2CS時
# def getArch_weights(arch, bins):
#     if arch == 'ResNet18':
#         model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
#         pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
#     elif arch == 'ResNet34':
#         model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
#         pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
#     elif arch == 'ResNet101':
#         model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
#         pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
#     elif arch == 'ResNet152':
#         model = L2CS(torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
#         pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
#     else:
#         if arch != 'ResNet50':
#             print('Invalid value for architecture is passed! '
#                   'The default value of ResNet50 will be used instead!')
#         model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
#         pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

#     return model, pre_url



#函數的主要作用是確定哪些參數需要被忽略，以便更好地控制模型的訓練過程。這樣可以提高模型訓練的靈活性和效率。
def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

#這個函數用於獲取模型中不需要被凍結的參數
#通常，在深度學習中，我們可能希望凍結某些層（例如，遷移學習中的預訓練層），以保持其權重不變，只優化特定層的權重。這個函數有助於確定要優化的參數。
def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


#這個函數用於獲取模型中的全連接層（Fully Connected Layer，即 FC 層）的參數。FC 層通常包含模型的最後一層，用於生成預測結果。
#在模型優化中，通常需要對這些 FC 層的參數進行優化，以便調整模型的預測能力。
def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param




#在模型載入時，有時候我們只想載入一部分權重（例如，從預訓練模型載入特定層的權重），而不是整個模型的權重。
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)



#建立dataset
dataset=datasets.Gaze360(gaze360label_dir, gaze360image_dir, transformations, 180, 4)
dataset_val=datasets.Gaze360(gaze360val_label_dir, gaze360image_dir, transformations, 180, 4)

#建立dataloader
train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=True)

val_loader_gaze = DataLoader(
            dataset=dataset_val,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=True)

#建立model


#L2CS 從頭訓練
# arch = 'ResNet50'
# model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90) 
# pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
# load_filtered_state_dict(model, model_zoo.load_url(pre_url))

#L2CS 有權重
# model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90) 
# path = 'output\snapshots\L2CS-gaze360-_1718167321\_epoch_13.pkl'
# saved_state = torch.load(path)
# model.load_state_dict(saved_state)

#ghostnet model 沒權重使用
# model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)  # 初始化您的模型
# pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
# load_filtered_state_dict(model, model_zoo.load_url(pre_url))

#ghostnet model 有權重使用
model = ResNet(Bottleneck, [3, 4, 6, 3], 90,s=2, d=3)
path = 'output\snapshots\L2CS-gaze360-_1718167321\_epoch_13.pkl'
saved_state = torch.load(path)
model.load_state_dict(saved_state)





#設定loss
criterion = nn.CrossEntropyLoss()
reg_criterion = nn.MSELoss()

#儲存
summary_name = '{}_{}'.format('L2CS-gaze360-', int(time.time()))
output=os.path.join(output, summary_name)
if not os.path.exists(output):
    os.makedirs(output)

#activaty function
softmax = nn.Softmax(dim=1)
idx_tensor = [idx for idx in range(90)]
idx_tensor = torch.FloatTensor(idx_tensor)



#設定optimizer Adam
optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model), 'lr': 0},
            {'params': get_non_ignored_params(model), 'lr': lr},
            {'params': get_fc_params(model), 'lr': lr}
        ], lr)



configuration = f"\ntrain configuration, gpu_id={gpu_id}, batch_size={batch_size}, model_arch={arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
print(configuration)

#adjust lr依照epoch數
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gaze, step_size=10, gamma=0.1)
#adjust lr依照val loss去調整的
#mode='min' 意味着调度器会在监测的指标停止减少时降低学习率；factor=0.1 表示新的学习率是原来的 0.1 倍；patience=5 意味着如果在 5 个 epoch 之后性能没有提升，就会降低学习率。
# scheduler = ReduceLROnPlateau(optimizer_gaze, mode='min', factor=0.1, patience=5, verbose=True)


model.cuda(gpu)

#紀錄loss
losses_pitch_gaze = []
losses_yaw_gaze = []
losses_val_pitch_gaze = []
losses_val_yaw_gaze = []
learning_rates = []

# 在訓練循環之前初始化早期停止所需的變數
# early_stopping_patience = 10
# best_val_loss = float('inf')
# epochs_no_improve = 0
# # 清理 CUDA 缓存
# torch.cuda.empty_cache()

#training
# scheduler = ReduceLROnPlateau(optimizer_gaze, 'min', patience=5, factor=0.3, verbose=True)

def main():


    for epoch in range(num_epochs):


        # model.train()  # 设置模型为训练模式
        sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
        
        for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
            images_gaze = Variable(images_gaze).cuda(gpu)
            
            # Binned labels
            label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
            label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

            # Continuous labels
            label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
            label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

            # optimizer_gaze.zero_grad()
            pitch, yaw = model(images_gaze)

            # Cross entropy loss
            loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
            loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

            # MSE loss
            pitch_predicted = softmax(pitch)
            yaw_predicted = softmax(yaw)

            #通过 softmax 和期望值计算，将分类预测的概率分布转换为实际的注视角度，并通过乘以4（区间宽度）和减去180度进行调整，以得到最终的预测值
            pitch_predicted = \
                torch.sum(pitch_predicted * idx_tensor.cuda(gpu), 1) * 4 - 180
            yaw_predicted = \
                torch.sum(yaw_predicted * idx_tensor.cuda(gpu), 1) * 4 - 180


            loss_reg_pitch = reg_criterion(
                pitch_predicted, label_pitch_cont_gaze)
            loss_reg_yaw = reg_criterion(
                yaw_predicted, label_yaw_cont_gaze)

            # Total loss
            loss_pitch_gaze += alpha * loss_reg_pitch
            loss_yaw_gaze += alpha * loss_reg_yaw

            

            sum_loss_pitch_gaze += loss_pitch_gaze
            sum_loss_yaw_gaze += loss_yaw_gaze
            

            loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
            grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

            optimizer_gaze.zero_grad(set_to_none=True)
            torch.autograd.backward(loss_seq, grad_seq)

            torch.cuda.empty_cache()
            optimizer_gaze.step()
            # ignored_params = list(get_ignored_params(model))

            # 清理 CUDA 缓存
            # torch.cuda.empty_cache()

            # scheduler.step()


            
            iter_gaze += 1

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                    'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                        epoch+1,
                        num_epochs,
                        i+1,
                        len(dataset)//batch_size,
                        sum_loss_pitch_gaze/iter_gaze,
                        sum_loss_yaw_gaze/iter_gaze
                    )
                    )
        # 每个 epoch 结束后清理 CUDA 缓存
        # torch.cuda.empty_cache()
        # scheduler.step()

        current_lrs = [group['lr'] for group in optimizer_gaze.param_groups]  # 获取当前学习率
        learning_rates.append(current_lrs)  # 记录学习率

        losses_pitch_gaze.append(sum_loss_pitch_gaze.item() / iter_gaze)
        losses_yaw_gaze.append(sum_loss_yaw_gaze.item() / iter_gaze)


        
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                torch.save(model.state_dict(),
                            output +'/'+
                            '_epoch_' + str(epoch+1) + '.pkl'))
            print(losses_pitch_gaze)
            print(losses_yaw_gaze)
            print(learning_rates)




        # Validation phase
        model.eval()  # Set model to evaluation mode
        sum_val_loss_pitch_gaze = sum_val_loss_yaw_gaze = iter_val_gaze = 0

        with torch.no_grad():  # No gradient tracking for validation
            for i, (images_gaze, labels_gaze, cont_labels_gaze, name) in enumerate(val_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)
                # Binned labels
                label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)
                # Continuous labels
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                pitch, yaw = model(images_gaze)

                # Calculate validation losses (similar to training)
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)
                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor.cuda(gpu), 1) * 4 - 180
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor.cuda(gpu), 1) * 4 - 180
                loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont_gaze)

                # Total validation loss
                sum_val_loss_pitch_gaze += (loss_pitch_gaze + alpha * loss_reg_pitch)
                sum_val_loss_yaw_gaze += (loss_yaw_gaze + alpha * loss_reg_yaw)
                iter_val_gaze += 1

        # 更新lr
        # scheduler.step(sum_val_loss_pitch_gaze/iter_val_gaze + sum_val_loss_yaw_gaze/iter_val_gaze)

        # Print validation losses
        print('Validation - Epoch [%d/%d] Losses: Gaze Yaw %.4f, Gaze Pitch %.4f' % (
            epoch+1,
            num_epochs,
            sum_val_loss_pitch_gaze/iter_val_gaze,
            sum_val_loss_yaw_gaze/iter_val_gaze
        ))
        losses_val_pitch_gaze.append(sum_val_loss_pitch_gaze.item() / iter_val_gaze)
        losses_val_yaw_gaze.append(sum_val_loss_yaw_gaze.item() / iter_val_gaze)
        print(losses_val_pitch_gaze)
        print(losses_val_yaw_gaze)

        #Early Stopping
        # val_loss = sum_val_loss_pitch_gaze.item() / iter_val_gaze
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1

        # if epochs_no_improve == early_stopping_patience:
        #     print('Early stopping triggered')
        #     break  # Break the training loop

# 繪製 loss 折線圖
def paint():
    plt.figure()
    plt.plot(losses_pitch_gaze, label='Train Gaze Pitch Loss')
    plt.plot(losses_yaw_gaze, label='Train Gaze Yaw Loss')
    plt.plot(losses_val_pitch_gaze, label='Val Gaze Pitch Loss')
    plt.plot(losses_val_yaw_gaze, label='Val Gaze Yaw Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output, 'loss_plot.png'))  # 將折線圖儲存為圖片

    plt.figure()
    for i, lrs in enumerate(zip(*learning_rates)):
        plt.plot(lrs, label=f'Group {i+1} LR')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig(os.path.join(output, 'learning_rate_plot.png'))

if __name__ == '__main__':
    main()
    paint()