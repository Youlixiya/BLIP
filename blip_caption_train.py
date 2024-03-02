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
from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers.models.blip.modeling_blip import BlipForConditionalGenerationModelOutput
from tensorboardX import SummaryWriter
from blip.model import BlipLDPNetV2ForConditionalGeneration

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def forward(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    labels: Optional[torch.LongTensor] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BlipForConditionalGenerationModelOutput]:
    r"""
    Returns:

    Examples:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, BlipForConditionalGeneration

    >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> text = "A picture of"

    >>> inputs = processor(images=image, text=text, return_tensors="pt")

    >>> outputs = model(**inputs)
    ```"""

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    vision_outputs = self.vision_model(
        pixel_values=pixel_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    image_embeds = vision_outputs[0][:, [0], :]

    outputs = self.text_decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        labels=labels,
        return_dict=return_dict,
        reduction="mean",
    )

    if not return_dict:
        outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
        return tuple(output for output in outputs if output is not None)

    return BlipForConditionalGenerationModelOutput(
        loss=outputs.loss,
        logits=outputs.logits,
        image_embeds=image_embeds,
        last_hidden_state=vision_outputs.last_hidden_state,
        hidden_states=vision_outputs.hidden_states,
        attentions=vision_outputs.attentions,
    )
BlipForConditionalGeneration.forward = forward

class BLIPDataset(Dataset):
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
        caption = self.datas[index]['conversations'][1]['value'] + '[SEP]'
        image = Image.open(image_path)
        # input = self.processor(images=image, text=caption, return_tensors="pt")
        return image, caption
        # except:
        #     self.__getitem__(0)
        
    
def collate_fn(batch, processor):
    # pixel_values = []
    # input_ids = []
    # attention_mask = []
    images = []
    captions = []
    # labels = []
    for image, caption in batch:
        images.append(image)
        captions.append(caption)
        # pixel_values.append(input['pixel_values'])
        # input_ids.append(input['input_ids'])
        # attention_mask.append(input['attention_mask'])
    # pixel_values = torch.cat(pixel_values)
    # input_ids = torch.cat(input_ids)
    # attention_mask = torch.cat(attention_mask)
    # labels = input_ids
    # inputs = {
    #     'pixel_values': pixel_values,
    #     'input_ids': input_ids,
    #     'attention_mask': attention_mask,
    #     'labels': labels
    # }
    inputs = processor(images=images,
                       text=captions,
                       return_tensors="pt",
                       padding=True,
                       truncation=True)
    inputs['labels'] = inputs['input_ids']
    return inputs

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
    parser.add_argument('--print_iters', type=int, default=50, help='print loss iterations')
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
    dtype = torch.float32
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
    model = BlipLDPNetV2ForConditionalGeneration.from_pretrained("./ckpts/blip-image-captioning-large")
    processor = AutoProcessor.from_pretrained("./ckpts/blip-image-captioning-large")
    train_dataset = BLIPDataset('data/sharegpt4v/subset-share-captioner_coco_lcs_sam_676k.json')
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, drop_last=True, collate_fn=partial(collate_fn, processor=processor))
    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))
    model.vision_model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
                    if key == 'pixel_values':
                        data[key] = value.to(device=args.local_rank, dtype=dtype)
                    else:
                        data[key] = value.to(device=args.local_rank)
            # data['return_loss'] = True
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
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLM Loss: {:.6f}'.format(
                        epoch, batch_idx * samples * dist.get_world_size(), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                    writer.add_scalar("LM loss", loss.item(), total_iters)
                
                # save model
                if total_iters % args.save_iters == 0:
                    # save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                    save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters))
                    print("save model to {}".format(save_path))
                    # torch.save(model.module.state_dict(), save_path)
                    model.module.save_pretrained(save_path)

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
        # torch.save(model.module.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        # model.module.save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        model.module.save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final"))
        writer.close()

    
    