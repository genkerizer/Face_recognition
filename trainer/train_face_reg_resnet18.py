import os
import yaml
import torch
import datetime

from torch.utils.data import DataLoader
from src.tools.data_utils import get_dist_info
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.loader.data_loader import DataLoaderX, DistributedSampler, MXFaceDataset
from src.models.backbones.mfacenet import MobileFaceNet
from src.models.backbones.resnet_18 import ResNet
from src.losses.partial_fc import PartialFC_V2
from src.losses.margin_loss import CombinedMarginLoss
from src.optimizers.scheduler import PolyScheduler
from src.tools.logging import AverageMeter


class Trainer:
    
    def __init__(self, config, **kwargs):
        self.global_config = config['Global']
        self.arch_config = config['Architecture']
        self.criterion_config = config['Loss']
        self.optimizer_config = config['Optimizer']
        self.data_config = config['Dataloader']
        self.checkpoint_config = config['Save']

        self.init_params()

        self.build_criterion()
        self.build_model()
        self.build_optimizer()
        self.build_dataloader()
    
        
        self.loss_am = AverageMeter()

        if self.global_config['use_pretrain']:
            print("LOAD WEIGHT")
            weight_path = "/content/drive/MyDrive/FACE_Project/CHECKPOINT_REG/checkpoint_step_45000.pt"
            weights = torch.load(weight_path)
            self.backbone.load_state_dict(weights['state_dict_backbone'])
            self.head.load_state_dict(weights['state_dict_softmax_fc'])
            self.optimizer.load_state_dict(weights['state_optimizer'])
            self.lr_scheduler.load_state_dict(weights['state_lr_scheduler'])
            print("DONE")


    def init_params(self):
        checkpoint_dir = self.global_config['checkpoints']
        self.ckpt_iter = os.path.join(checkpoint_dir, "ITER")
        self.ckpt_epoch = os.path.join(checkpoint_dir, "EPOCH")
        os.makedirs(self.ckpt_iter, exist_ok=True)
        os.makedirs(self.ckpt_epoch, exist_ok=True)

        self.save_iter = self.checkpoint_config['save_iter']
        self.num_image = self.data_config['num_image']
        self.total_batch_size = self.data_config['batch_size']
        self.warmup_epoch = self.global_config['warmup_epoch']
        self.num_epoch = self.global_config['num_epoch']
        self.warmup_step = self.num_image // self.total_batch_size * self.warmup_epoch
        self.total_step = self.num_image // self.total_batch_size * self.num_epoch


    def build_model(self):
        self.backbone = ResNet(**self.arch_config['Backbone']).cuda()
        self.head = PartialFC_V2(self.criterion, **self.arch_config['Head']).cuda()


    def get_backbone(self, checkpoint_path):
        print("Get backbone")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        backbone = checkpoint['state_dict_backbone']
        torch.save(backbone, "face_reg_torch/models/backbone_weight.pt")
        print("DONE")

    def build_criterion(self):
        self.criterion = CombinedMarginLoss(64,
                                            self.criterion_config['margin_list'][0],
                                            self.criterion_config['margin_list'][1],
                                            self.criterion_config['margin_list'][2],
                                            self.criterion_config['interclass_filtering_threshold']
                                    ) 

    def build_optimizer(self):

        lr = self.optimizer_config['lr']
        weight_decay = self.optimizer_config['weight_decay']
        self.optimizer = torch.optim.AdamW(
            params=[{"params": self.backbone.parameters()}, {"params": self.head.parameters()}],
            lr=lr, weight_decay=weight_decay)

        self.lr_scheduler = PolyScheduler(optimizer=self.optimizer,
                                          base_lr=lr,
                                          max_steps=self.total_step,
                                          warmup_steps=self.warmup_step,
                                          last_epoch=self.optimizer_config['last_epoch'])
        




    def build_dataloader(self):
        root_dir = self.data_config['root_dir']
        local_rank = 0
        batch_size = self.total_batch_size
        dali = False
        seed = 2048
        num_workers = 0

        rec = os.path.join(root_dir, 'train.rec')
        idx = os.path.join(root_dir, 'train.idx')
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)


        # DALI
      
        rank, world_size = get_dist_info()
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

        self.train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=None,
        )


    def train(self):
        start_epoch = 1
        global_step = 0
        gradient_acc = self.optimizer_config['gradient_acc']
        

        for epoch in range(start_epoch, self.num_epoch + 1):
            if isinstance(self.train_loader, DataLoader):
                self.train_loader.sampler.set_epoch(epoch)

            for _, (img, local_labels) in enumerate(self.train_loader):
                global_step += 1
                local_embeddings = self.backbone(img)
                loss: torch.Tensor = self.head(local_embeddings, local_labels)

                loss.backward()
                if global_step % gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.lr_scheduler.step()

                if global_step % 2000 == 0 and global_step > 0:
                    with torch.no_grad():
                       print({'Process/Epoch': epoch, 'Process/Step': global_step, 'Loss/Step Loss': loss.item(),'Loss/Train Loss': self.loss_am.avg})
                    

                if global_step % self.save_iter == 0 and global_step > 0:
                    checkpoint = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "state_dict_backbone": self.backbone.state_dict(),
                        "state_dict_softmax_fc": self.head.state_dict(),
                        "state_optimizer": self.optimizer.state_dict(),
                        "state_lr_scheduler": self.lr_scheduler.state_dict()
                    }
                    torch.save(checkpoint, os.path.join(self.ckpt_iter, f"checkpoint_step_{global_step}.pt"))

            
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "state_dict_backbone": self.backbone.state_dict(),
                "state_dict_softmax_fc": self.head.state_dict(),
                "state_optimizer": self.optimizer.state_dict(),
                "state_lr_scheduler": self.lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(self.ckpt_epoch, f"checkpoint_epoch_{epoch}.pt"))

        return None


if __name__ == '__main__':
    config = yaml.load(open('configs/face_reg_resnet18.yml'))
    trainer = Trainer(config)
    trainer.train()