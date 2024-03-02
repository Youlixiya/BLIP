import os
import json
import random
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from typing import Any, Optional, Tuple, Union
from functools import partial
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup, AutoProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers.models.blip.modeling_blip import BlipForConditionalGenerationModelOutput
from tensorboardX import SummaryWriter
from blip.model import BlipLDPNetV2ForConditionalGeneration, BlipVisionAeForQuestionAnswering

from torch import distributed as dist
# from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class BLIPQADataset(Dataset):
    def __init__(self,
                 data_path):
        super().__init__()
        self.datas = json.load(open(data_path))
        # self.processor = AutoProcessor.from_pretrained("./ckpts/blip-image-captioning-large")
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        # try:
        image_path = os.path.join('data', self.datas[index]['image'])
        # caption = self.datas[index]['caption'] + ' [SEP]'
        question = self.datas[index]['question']
        answer = self.datas[index]['answer']
        image = Image.open(image_path)
        # input = self.processor(images=image, text=caption, return_tensors="pt")
        return image, question, answer
        # except:
        #     self.__getitem__(0)
        
    
def collate_fn(batch, processor):

    images = []
    questions = []
    answers = []

    for image, question, answer in batch:
        images.append(image)
        questions.append(question)
        answers.append(answer)
        
    
    inputs = processor(images=images,
                       text=questions,
                       return_tensors="pt",
                       padding=True,
                       truncation=True)
    labels = processor(text=answers,
                       return_tensors="pt",
                       padding=True,
                       truncation=True).input_ids
    inputs['labels'] = labels
    return inputs

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data/sa/images", help='root path of dataset')

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
    parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='gradient accumulation steps')
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
    parser.add_argument('--save_iters', type=int, default=10000, help='save iterations')

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
    accelerator = Accelerator()
    device = accelerator.device
    # device = torch.device('cuda', args.local_rank)
    # dtype = torch.float32
    # torch.distributed.init_process_group(backend='nccl')
    
    # file folder creating
    # if args.local_rank == 0:
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
    model = BlipForQuestionAnswering.from_pretrained("./ckpts/blip-vqa-capfilt-large")
    processor = AutoProcessor.from_pretrained("./ckpts/blip-vqa-capfilt-large")
    # processor.tokenizer.add_tokens(['<ref>', '</ref>', '<box>', '</box>'], special_tokens=True)
    # model.resize_token_embeddings(len(processor.tokenizer))
    train_dataset = BLIPQADataset('data/blip/blip_refcoco3_rec_127k.json')
    # train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=partial(collate_fn, processor=processor))
    # if args.local_rank == 0:
        # writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))
    model.vision_model.requires_grad_(False)
    
    # model.to(device=device, dtype=dtype)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
     # optimizer and scheduler
    # optimizer = get_optimizer(args, model.module.vision_model)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_loader) * args.epochs) // args.gradient_accumulation_steps,
    )
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader)

    total_iters = 0
    
    for epoch in range(1, args.epochs + 1):
        # new epoch
        # if args.local_rank == 0:
        accelerator.print("------start epoch {}------".format(epoch))
        # train_sampler.set_epoch(epoch)
        # training
        model.train()
        for batch_idx, data in enumerate(train_loader):
            total_iters += 1
            samples = data['pixel_values'].shape[0]
            for key, value in data.items():
                if type(value) == torch.Tensor:
                    if key == 'pixel_values':
                        data[key] = value.to(device=device)
                    else:
                        data[key] = value.to(device=device)
            # data['return_loss'] = True
            optimizer.zero_grad()
            output = model(**data)
            loss = output.loss
            # print(loss.item())
            # loss = reduce_mean(output.loss, dist.get_world_size())
            # loss.backward()
            accelerator.backward(loss)
            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accelerator.print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLM Loss: {:.6f}'.format(
                    epoch, batch_idx // args.gradient_accumulation_steps, len(train_loader) // args.gradient_accumulation_steps,
                        100. * batch_idx / len(train_loader), loss.item()))
            # optimizer.step()
            
            # if is master process
            # if args.local_rank == 0:
            # print training info
            # if (batch_idx + 1) % args.print_iters == 0:
                # accelerator.print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLM Loss: {:.6f}'.format(
                #     epoch, batch_idx * samples * dist.get_world_size(), len(train_loader.dataset),
                #         100. * batch_idx / len(train_loader), loss.item()))
                # writer.add_scalar("LM loss", loss.item(), total_iters)
            
            # save model
            if total_iters % args.save_iters == 0:
                # save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters))
                accelerator.print("save model to {}".format(save_path))
                # torch.save(model.module.state_dict(), save_path)
                accelerator.unwrap_model(model).save_pretrained(save_path)
                processor.save_pretrained(save_path)

                # evaluation
                '''
                if total_iters % args.eval_iters == 0:
                    test_loss = test(args, model, val_loader)
                    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
                    writer.add_scalar("eval_mse_loss", test_loss, total_iters)
                '''

        # dist.barrier()
        # scheduler.step()

    # save final model
    # if args.local_rank == 0:
        # torch.save(model.module.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        # model.module.save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
    accelerator.unwrap_model(model).save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final"))
    processor.save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final"))
    # writer.close()

    
    