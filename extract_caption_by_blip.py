import os
import json
import random
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from tensorboardX import SummaryWriter

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class SAMImageDataset(Dataset):

    def __init__(self,
                 image_dir,
                 ):
        self.image_file_list = os.listdir(image_dir)
        self.image_path_list = [os.path.join(image_dir, i) for i in self.image_file_list]
        # self.image_list = [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in self.image_path_list]
        self.processor = AutoProcessor.from_pretrained("./ckpts/blip-image-captioning-large")
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        # caption_save_path = self.image_path_list[idx].replace('images', 'captions').replace('jpg', 'pt')
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.pixel_values, image_path
        
    
def collate_fn(batch):
    pixel_values = []
    image_paths = []
    for pixel_value, image_path in batch:
        pixel_values.append(pixel_value)
        image_paths.append(image_path)

    return torch.cat(pixel_values), image_paths

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data/sam/images", help='root path of dataset')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--ckpt', type=str, default='', help='model pretrained ckpt')

    # multi gpu settings
    parser.add_argument("--local-rank", type=int, default=-1)

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    # learning process settings
    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=1, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default=".", help='root path')
    parser.add_argument('--work_dir', type=str, default="work_dir", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpts", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=50000, help='save iterations')

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_option()
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    
    # file folder creating
    if args.local_rank == 0:
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))
        
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark
    model = BlipForConditionalGeneration.from_pretrained("./ckpts/blip-image-captioning-large")
    # config = CLIP3DTextConfig()
    # model.requires_grad_(False)
    train_dataset = SAMImageDataset(args.dataset_path)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, sampler=train_sampler, collate_fn=collate_fn)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    total_iters = 0
    captions_json = []
    for epoch in range(1, args.epochs + 1):
        # training
        model.train()
        for pixel_values, image_paths in tqdm(train_loader):
            total_iters += 1
            pixel_values = pixel_values.cuda(args.local_rank)
            data = {'pixel_values': pixel_values}
            output = model.module.generate(**data)
            tmp_dict = {}
            tmp_dict['image'] = image_paths[0]
            tmp_dict['caption'] = train_dataset.processor.decode(output[0], skip_special_tokens=True)
            # print(tmp_dict)
            captions_json.append(tmp_dict)
            
        dist.barrier()
    if args.local_rank == 0:
        with open('data/sam/blip_captions.json', 'w') as f:
            json.dump(captions_json, f)

    
    