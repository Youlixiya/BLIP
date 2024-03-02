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
from transformers import CLIPTextModel, AutoTokenizer, Trainer, TrainingArguments
from clip3d import CLIP3DModel, CLIP3DConfig
from tensorboardX import SummaryWriter

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def voxel_preprocess(voxel, resolution=160):
    ndim = voxel.ndim
    if voxel.ndim == 4:
        voxel = voxel[None]
    x, y, z = voxel.shape[2:]
    scale_x, scale_y, scale_z = resolution / x, resolution / y, resolution / z
    scale = min(scale_x, scale_y, scale_z)
    new_x, new_y, new_z = int(scale * x), int(scale * y), int(scale * z)
    # print(new_x, new_y, new_z)
    pad_x, pad_y, pad_z = resolution - new_x, resolution - new_y, resolution - new_z
    # print(pad_x, pad_y, pad_z)
    voxel = torch.nn.functional.interpolate(voxel, scale_factor=scale)
    voxel = torch.nn.functional.pad(voxel, (0, pad_z, 0, pad_y, 0, pad_x))
    if ndim == 4:
        voxel = voxel.squeeze(0)
    return voxel

class CLIP3DDataset(Dataset):
    def __init__(self,
                 feature_root,
                 caption_path,
                 resolution=160) -> None:
        super().__init__()
        self.feature_data_file_names = os.listdir(feature_root)
        self.caption_json = json.load(open(caption_path))
        feature_data_names = [feature_data_file_name.split('.')[0] for feature_data_file_name in self.feature_data_file_names]
        self.feature_data_names = [feature_data_name for feature_data_name in feature_data_names if feature_data_name in self.caption_json.keys()]
        self.feature_data_paths = [os.path.join(feature_root, feature_data_file_name) for feature_data_file_name in self.feature_data_file_names]
        
        self.voxel_preprocess = partial(voxel_preprocess, resolution=resolution)
    def __len__(self):
        return len(self.feature_data_names)
    
    def __getitem__(self, index):
        # try:
        feature_data_name = self.feature_data_names[index]
        feature_data_path = self.feature_data_paths[index]
        feature = torch.from_numpy(np.load(feature_data_path)['rgbsigma']).permute(3, 0, 1, 2)
        caption = self.caption_json[feature_data_name]['caption']
        return self.voxel_preprocess(feature), caption
        # except:
        #     self.__getitem__(0)
        
    
def collate_fn(batch, tokenizer):
    features = []
    captions = []
    for feature, caption in batch:
        features.append(feature)
        captions.append(caption)
    output =  tokenizer(captions,
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=77)
    # print(output)
    # print(torch.stack(features).shape)
    # output['input_ids'] = torch.LongTensor(output['input_ids'])
    output['pixel_values'] = torch.stack(features)
    return output

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data/sa/images", help='root path of dataset')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=8, help='number of training epochs')
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

def get_optimizer(args, model):
    if args.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)

def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
    
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
    tokenizer = AutoTokenizer.from_pretrained("./ckpts/clip-vit-base-patch32")
    # config = CLIP3DTextConfig()
    config = CLIP3DConfig()
    # clip3d_text_model = CLIP3DTextModel(config=config)
    clip3d = CLIP3DModel(config=config)
    clip_text_model = CLIPTextModel.from_pretrained('./ckpts/clip-vit-base-patch32')
    clip3d.text_model.load_state_dict(clip_text_model.text_model.state_dict())
    del clip_text_model
    clip3d.text_model.requires_grad_(False)
    train_dataset = CLIP3DDataset('data/nerf_datasets/hypersim_rpn_data/features',
                            'data/nerf_datasets/hypersim_nerf_scene_captions_data.json')
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, drop_last=True, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))
    clip3d.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(clip3d)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
     # optimizer and scheduler
    # optimizer = get_optimizer(args, model.module.vision_model)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    total_iters = 0
    
    for epoch in range(1, args.epochs + 1):
        # new epoch
        if args.local_rank == 0:
            print("------start epoch {}------".format(epoch))
        train_sampler.set_epoch(epoch)

        # training
        model.train()
        for batch_idx, data in enumerate(train_loader):
            total_iters += 1
            samples = data['pixel_values'].shape[0]
            for key, value in data.items():
                if type(value) == torch.Tensor:
                    data[key] = value.cuda(args.local_rank)
            data['return_loss'] = True
            optimizer.zero_grad()
            output = model(**data)
            # loss = output.loss
            # print(loss.item())
            loss = reduce_mean(output.loss, dist.get_world_size())
            loss.backward()
            optimizer.step()
            
            # if is master process
            if args.local_rank == 0:
                # print training info
                if (batch_idx + 1) % args.print_iters == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                        epoch, batch_idx * samples * dist.get_world_size(), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                    writer.add_scalar("loss", loss.item(), total_iters)
                
                # save model
                if total_iters % args.save_iters == 0:
                    save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                    print("save model to {}".format(save_path))
                    torch.save(model.module.state_dict(), save_path)

                # evaluation
                '''
                if total_iters % args.eval_iters == 0:
                    test_loss = test(args, model, val_loader)
                    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
                    writer.add_scalar("eval_mse_loss", test_loss, total_iters)
                '''

        dist.barrier()
        scheduler.step()

    # save final model
    if args.local_rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        writer.close()

    
    