import torch
from utils import select_device, draw_gaze
from model.L2CS_MobileOne import L2CS



model = L2CS(90)
print('Loading snapshot.')
saved_state_dict = torch.load('output/snapshots/L2CS-gaze360-_1705977930/_epoch_29.pkl')
model.load_state_dict(saved_state_dict)
model.cuda(0)
model.eval()
# Display all model layer weights
for name, para in model.named_parameters():
    print('{}: {}'.format(name, para.shape))