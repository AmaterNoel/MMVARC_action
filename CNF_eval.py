import time
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from torchmetrics.classification import MultilabelAveragePrecision

from post_model import *
from dataset import *

parser = argparse.ArgumentParser(description='Double reasoning')
parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--device", default='cuda:5', type=str, help="0, 1, 2, 3, 4, 5, 6, 7")
parser.add_argument("--type", default='overall', type=str, help="overall, head_type, middle_type, tail_type")
parser.add_argument("--clip_pt", default='/mnt/gemlab_data/User_database/longnuoer/MMVARC/pre_model/text_embed_none/stage2', help="Address of image-text matching model parameters, Default: path or None")
parser.add_argument("--pre_pt", default='/mnt/gemlab_data/User_database/longnuoer/MMVARC/post_model/text_embed_none/stage3', help="Load the trained model, Default: path or None")
parser.add_argument("--frames", default=8, type=int, help="Number of frames in a video")
parser.add_argument("--batch_size", default=32, type=int, help="Size of the mini-batch")
parser.add_argument('--dataroot', default=r'/mnt/sm870/lne/AnimalKingdom', type=str, help='dataset root')
opt = parser.parse_args()

# define dataset
action_class_list, animal_class_list = get_dict()
action_class_list = list(action_class_list.keys())
print("num_action_class_list = ", len(action_class_list))

eval_set = AnimalKingdom_action_eval(root=opt.dataroot, type=opt.type, total_length=opt.frames, action_class_list=action_class_list, random_shift=False)
eval_sampler = torch.utils.data.RandomSampler(eval_set)
eval_loader = torch.utils.data.DataLoader(
    dataset=eval_set,
    batch_size=opt.batch_size,
    sampler=eval_sampler,
    shuffle=False
)

# define post_model
text_embed = get_text_features(action_class_list)
text_embed = F.adaptive_avg_pool1d(text_embed.unsqueeze(0), 256).squeeze(0)
device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
text_embed = text_embed.to(device)
post_model = TSDETR(text_embed=text_embed,num_frames=opt.frames, num_queries=len(action_class_list), clip_pt=opt.clip_pt, device=device).to(device)
if opt.pre_pt != 'None':
    post_model.load_state_dict(torch.load(opt.pre_pt, map_location=device))
for p in post_model.parameters():
    p.requires_grad = True
for p in post_model.pre_model.parameters():
    p.requires_grad = False
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(post_model.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, 200)
action_eval_metric = MultilabelAveragePrecision(num_labels=len(action_class_list), average='micro')
print('Parameter Space: ABS: {:.2f}'.format(count_parameters(post_model) / 268435456) + " GB")

# define eval
post_model.eval()
action_loss_meter = AverageMeter()
for data, action_label in tqdm(eval_loader):
    data = data.to(device)
    action_label = action_label.long().to(device)

    with torch.no_grad():
        action_pred = post_model(data)
    action_eval = action_eval_metric(action_pred, action_label)
    action_loss_meter.update(action_eval.item(), data.shape[0])
print("[INFO] Action Evaluation Metric: {:.2f}".format(action_loss_meter.avg * 100), flush=True)