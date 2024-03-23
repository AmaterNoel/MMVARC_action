import time
import torch
import argparse
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAveragePrecision

from pre_model import *
from dataset import *

parser = argparse.ArgumentParser(description='Double reasoning')
parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--device", default='cuda:6', type=str, help="0, 1, 2, 3, 4, 5, 6, 7")
parser.add_argument("--pre_pt", default='/mnt/gemlab_data/User_database/longnuoer/MMVARC/pre_model/text_embed_clip/stage3', help="Load the trained model, Default: path or None")
parser.add_argument("--save_pt", default='/mnt/gemlab_data/User_database/longnuoer/MMVARC/pre_model/text_embed_clip/stage4', help="The address where the model is stored after training is completed, Default: path or None")
parser.add_argument("--total_epoch", default=200, type=int, help="epoch")
parser.add_argument("--frames", default=8, type=int, help="Number of frames in a video")
parser.add_argument("--batch_size", default=32, type=int, help="Size of the mini-batch")
parser.add_argument("--test_every", default=5, type=int, help="Test the model every this number of epochs")
parser.add_argument('--dataroot', default=r'/mnt/sm870/lne/AnimalKingdom', type=str, help='dataset root')
opt = parser.parse_args()

# define dataset
action_class_list, animal_class_list = get_dict()
action_class_list = list(action_class_list.keys())
print("num_action_class_list = ", len(action_class_list))

train_set = AnimalKingdom_action(root=opt.dataroot, type='train', total_length=opt.frames, action_class_list=action_class_list, random_shift=False)
train_sampler = torch.utils.data.RandomSampler(train_set, num_samples=2500)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=opt.batch_size,
    sampler=train_sampler,
    shuffle=False
)

test_set = AnimalKingdom_action(root=opt.dataroot, type='test', total_length=opt.frames, action_class_list=action_class_list, random_shift=False)
test_sampler = torch.utils.data.RandomSampler(test_set)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=opt.batch_size,
    sampler=test_sampler,
    shuffle=False
)

# define pre_model
text_embed = get_text_features(action_class_list)
text_embed = F.adaptive_avg_pool1d(text_embed.unsqueeze(0), 256).squeeze(0)

device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
pre_model = Action_CLIP(text_embed=text_embed, num_queries=len(action_class_list), device=device).to(device)
if opt.pre_pt != 'None':
    pre_model.load_state_dict(torch.load(opt.pre_pt, map_location=device))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(pre_model.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, 200)
action_eval_metric = MultilabelAveragePrecision(num_labels=len(action_class_list), average='micro')
print('Parameter Space: ABS: {:.2f}'.format(count_parameters(pre_model) / 268435456) + " GB")

# define train
for index in tqdm(range(opt.total_epoch)):
    pre_model.train()
    action_loss_meter = AverageMeter()
    start_time = time.time()
    for data, action_label in train_loader:
        data = data.to(device)
        action_label = action_label.to(device)

        optimizer.zero_grad()
        action_pred = pre_model(data)

        action_loss = criterion(action_pred, action_label)
        action_loss.backward()
        optimizer.step()

        action_loss_meter.update(action_loss, data.shape[0])
    elapsed_time = time.time() - start_time
    scheduler.step()
    print("Epoch [" + str(index + 1) + "]"
          + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
          + " action_loss: " + "{:.4f}".format(action_loss_meter.avg), flush=True)

    if (index + 1) % opt.test_every == 0:
        pre_model.eval()
        action_loss_meter = AverageMeter()
        for data, action_label in test_loader:
            data = data.to(device)
            action_label = action_label.long().to(device)

            with torch.no_grad():
                action_pred = pre_model(data)
            action_eval = action_eval_metric(action_pred, action_label)
            action_loss_meter.update(action_eval.item(), data.shape[0])
        print("[INFO] Action Evaluation Metric: {:.2f}".format(action_loss_meter.avg * 100), flush=True)

torch.save(pre_model.state_dict(), opt.save_pt)