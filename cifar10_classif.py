import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torchvision import transforms, datasets
from utils.util import AverageMeter, accuracy, Identity
from utils.loses import SupConLoss
from data.cifarloader import CIFAR10Loader
from models.resnet_3x3 import ResNet, BasicBlock
import os
import sys
import time
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def train(model, train_loader, eval_loader, unlabeled_eval_loader, save_path, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion=SupConLoss().cuda(device)
    train_losses = []
    eval_losses = []

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        
        model.train()
        exp_lr_scheduler.step()

        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            images = torch.cat([x, x_bar], dim=0)
            images, target = images.to(device), label.to(device)
            bsz = target.size(0)

            #calculate loss
            optimizer.zero_grad()
            features = model(images)

            print(features.shape)
            print(features)

            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            print(features.shape)
            print(features)
            print(target.shape)
            print(target)
            loss = criterion(features, target)
            
            # update metric
            loss_record.update(loss.item(), bsz)

            # update model
            loss.backward()
            optimizer.step()
            break

            
        print('Train Epoch: {} Avg Loss: {:.4f} \t '.format(epoch, loss_record.avg))
        avg_val_loss = test(model, eval_loader, args)

        train_losses.append(loss_record.avg)
        eval_losses.append(avg_val_loss)

        break

        if (epoch+1) % 2 == 0:
            plot_features(model, unlabeled_eval_loader, save_path, epoch, device, args)

            epoch_model_path = os.path.join(model_dir, f'{args.model_name}_epoch{epoch + 1}.pth')
            save_model(model, optimizer, exp_lr_scheduler, epoch+1, epoch_model_path)
        

    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

    plot_loss(train_losses, eval_losses, save_path)



def test(model, test_loader, args):
    model.eval()
    loss_record = AverageMeter()
    criterion=SupConLoss().cuda(device)


    for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(test_loader)):
        images = torch.cat([x, x_bar], dim=0)
        images, target = images.to(device), label.to(device)
        bsz = target.size(0)
        
        #calculate loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, target)
        
        # update metric
        loss_record.update(loss.item(), bsz)

    print('Test: Avg loss: {:.4f}'.format(loss_record.avg))
    return loss_record.avg

def save_model(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, path)

    print(f"Model saved to {path}")

def load_model(model, optimizer, scheduler, path, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded model from {path}, starting from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {path}, starting from scratch")
        start_epoch = 0
    return model, optimizer, scheduler, start_epoch


def plot_features(model, test_loader, save_path, epoch, device,  args):
    model.eval()
    
    targets=np.array([])
    outputs = np.zeros((len(test_loader.dataset), 
                      512 * BasicBlock.expansion)) 
    
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        output = model(x)
       
        outputs[idx, :] = output.cpu().detach().numpy()
        targets=np.append(targets, label.cpu().numpy())

   
    
    # print('plotting t-SNE ...') 
    # tsne plot
     # Create t-SNE visualization
    X_embedded = TSNE(n_components=2).fit_transform(outputs)  # Use meaningful features for t-SNE

    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=targets, cmap='viridis')
    plt.title("t-SNE Visualization of unlabeled Features on " + args.dataset_name + " unlabelled set - epoch" + str(epoch))
    plt.savefig(save_path+ '/' + args.dataset_name + '_epoch'+ str(epoch) + '.png')

def plot_loss(tr_loss, val_loss, save_path):
    plt.figure()
    plt.plot(tr_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_path + '/loss_plot.png')
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cls',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', default=5, type=int) # 180
    parser.add_argument('--milestones', default=[100, 150], type=int, nargs='+')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='resnet18_cifar10_classif_5')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    runner_name = os.path.basename(__file__).split(".")[0] 
    model_dir= args.exp_root + '{}'.format(runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'.pth'


    train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',labeled = True, aug='twiceCrop', shuffle=True)
    eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', labeled = True, aug='twiceCrop', shuffle=False)

    unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', labeled=False, aug=None, shuffle=False)

    model = ResNet(BasicBlock, [2,2,2,2], args.num_classes).to(device)
    model.linear = Identity()

    train(model, train_loader, eval_loader, unlabeled_eval_loader, model_dir, args)
    test(model, eval_loader, args)
