import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np
import cv2 as cv
import random
import time
import os
from models.models import EncoderRNN, load_partial_dict
from data.moving_mnist import MovingMNIST
from data.sst import SST
from data.texibj import TexiBJ
from data.human import Human
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
# from tools import load_partial_dict

milestones = [99, 350, 600]


def train(args):

    args.world_size = args.gpus * args.nodes  #
    print('world size ', args.world_size)
    # os.environ['MASTER_ADDR'] = 'localhost'  #
    # os.environ['MASTER_PORT'] = '8888'  #
    if args.world_size == 1:
        trainIters(0, args)
    else:
        mp.spawn(trainIters, nprocs=args.gpus, args=(args,))


def test(args):

    # select channel and size of input frames
    if args.dataset == 'mm':
        channel = 1
        size = (64, 64)
        dataset = MovingMNIST(root=args.root, is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])
    elif args.dataset == 'texibj':
        channel = 2
        size = (32, 32)
        dataset = TexiBJ(root=args.root, is_train=False)
    elif args.dataset == 'human':
        channel = 3
        size = (64, 64)
        dataset = Human(root=args.root, is_train=False)
    else:
        # sst
        channel = 1
        size = (64, 64)
        dataset = SST(root=args.root, is_train=False)

    encoder = EncoderRNN(channel, size)
    loc = 'cuda:0'
    encoder = encoder.to(device)
    if args.model and os.path.exists(args.model):
        print('===> load existing model: ', args.model)
        ckpt = torch.load(args.model, map_location=loc)
        encoder.load_state_dict(ckpt['phydnet'])
        # encoder.load_state_dict(ckpt)
    # encoder.eval()


    if args.eval:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)
        evaluate(encoder, loader, args)

    if args.visual:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
        visualize(encoder, loader, dataset=args.dataset)


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion,
                   teacher_forcing_ratio, args):
    '''

    :param input_tensor:
    :param target_tensor:
    :param encoder:
    :param encoder_optimizer:
    :param criterion:
    :param teacher_forcing_ratio:
    :param scaler:
    :param args:
    :return:
    '''
    
    torch.autograd.set_detect_anomaly(True)
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length  = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0
    hidden1 = None
    hidden2 = None
    ex_input = None
    # local = None

    for ei in range(input_length - 1):
        encoder_output, ex_input, output_image, hidden1, hidden2 = encoder(input=input_tensor[:, ei, :, :, :],
                                                                           ex_input_conv=ex_input,
                                                                           hidden1=hidden1,
                                                                           hidden2=hidden2,
                                                                           timestep=ei)

        # encoder_output, ex_input, output_image, hidden1, hidden2 = encoder(input=input_tensor[:, ei, :, :, :],
        #                                                                    ex_input_conv=ex_input,
        #                                                                    hidden1=hidden1,
        #                                                                    hidden2=hidden2,
        #                                                                    timestep=ei,
        #                                                                    local=local)

        if ei > 0:
            # calculate loss starting from the 6th frame
            loss = loss + criterion(output_image, input_tensor[:, ei + 1, :, :, :])

    decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input = last image of input sequence

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    for di in range(target_length):

        decoder_output, ex_input, output_image, hidden1, hidden2 = encoder(input=decoder_input,
                                                                           ex_input_conv=ex_input,
                                                                           hidden1=hidden1,
                                                                           hidden2=hidden2,
                                                                           timestep=di + input_length)

        # decoder_output, ex_input, output_image, hidden1, hidden2 = encoder(input=decoder_input,
        #                                                                    ex_input_conv=ex_input,
        #                                                                    hidden1=hidden1,
        #                                                                    hidden2=hidden2,
        #                                                                    timestep=di + input_length,
        #                                                                    local=local)

        target = target_tensor[:, di, :, :, :]
        loss = loss + criterion(output_image, target)
        if use_teacher_forcing:
            decoder_input = target  # Teacher forcing
        else:
            decoder_input = output_image

    dist.barrier()
    loss.backward()
    encoder_optimizer.step()
    # print('good')
    return loss.item() / target_length


def trainIters(gpu, args):
    '''

    :param gpu: the rank of  current gpu in this node
    :param encoder: an instance of the network
    :param nepochs: number of epochs, int
    :param print_every: how many epochs between 2 print messages, int
    :param eval_every: how many epochs between 2 evaluation processes, int
    :param name: the name of output model, str
    :param model_path: the path of existing model, str
    :return:
    '''
    # load parameter from args
    nepochs = args.nepochs
    print_every = args.print_every
    eval_every = args.eval_every
    name = args.save_name
    model_path = args.model
    gen_model = args.pre_model
    rank = args.nr * args.gpus + gpu

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    # select channel and size of input frames
    if args.dataset == 'mm' or args.dataset == 'sst':
        channel = 1
        size = (64, 64)
    elif args.dataset == 'texibj':
        channel = 2
        size = (32, 32)
    else:
        channel = 3
        size = (64, 64)

    encoder  = EncoderRNN(channel, size)

    train_losses = []
    start = 0

    # tensorboard
    if rank == 0:
        writer = SummaryWriter()

    # initialize DDP
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:23456',
        world_size=args.world_size,
        rank=rank
    )

    encoder.cuda(gpu)
    # encoder = torch.nn.DataParallel(encoder)
    # encoder.to(device)

    # loss
    criterion = nn.MSELoss().cuda(gpu)
    if torch.cuda.device_count() > 1:
        print('using {}/{}  gpus'.format(gpu, torch.cuda.device_count()))
        # encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        # encoder = DDP(encoder, device_ids=[rank], find_unused_parameters=True)
        encoder = DDP(encoder, device_ids=[rank])


    # load last model
    loc = 'cuda:{}'.format(gpu)
    if torch.cuda.device_count() > 1:
        print('multi-gpu')
        if model_path and os.path.exists(model_path):
            print('===> load existing model: ', model_path)
            ckpt = torch.load(args.model, map_location=loc)
            encoder.module.load_state_dict(ckpt['phydnet'])
            start = ckpt['epoch'] + 1
            print('start from ', start)
            # optimizer
            policies = encoder.module.get_optim_policies()
            for param_group in policies:
                param_group['lr'] = args.lr * param_group['lr_mult']
            encoder_optimizer = torch.optim.Adam(policies)

        elif gen_model or args.r2d_model != 'None':
            print(gen_model)
            print(args.r2d_model)
            load_partial_dict(net_ckpt=gen_model, model=encoder.module, loc=loc,
                              layer_ckpt=args.r2d_model)
            # param_group
            policies = encoder.module.get_optim_policies()
            for param_group in policies:
                param_group['lr'] = args.lr * param_group['lr_mult']
            encoder_optimizer = torch.optim.Adam(policies)

        else:
            policies = None
            encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    else:
        print('single gpu')
        if model_path and os.path.exists(model_path):
            print('===> load existing model: ', model_path)
            ckpt = torch.load(args.model, map_location=loc)
            encoder.load_state_dict(ckpt['phydnet'])
            start = ckpt['epoch'] + 1
            print('start from ', start)
            # optimizer
            policies = encoder.get_optim_policies()
            for param_group in policies:
                param_group['lr'] = args.lr * param_group['lr_mult']
            encoder_optimizer = torch.optim.Adam(policies)
        
        elif gen_model or args.r2d_model != 'None':
            print(gen_model)
            print(args.r2d_model)
            load_partial_dict(net_ckpt=gen_model, model=encoder, loc=loc,
                              layer_ckpt=args.r2d_model)
            # param_group
            policies = encoder.get_optim_policies()
            for param_group in policies:
                param_group['lr'] = args.lr * param_group['lr_mult']
            encoder_optimizer = torch.optim.Adam(policies)

        else:
            policies = None
            encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    # load state of optimizer
    if model_path and os.path.exists(model_path):
        encoder_optimizer.load_state_dict(ckpt['optimizer'])
    scheduler_rop = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=2,factor=0.9,verbose=True)
    scheduler_mstep = MultiStepLR(encoder_optimizer, milestones, gamma=0.9, verbose=True)

    # dataset
    if args.dataset == 'mm':
        dataset_train = MovingMNIST(root=args.root, is_train=True, n_frames_input=10,
                                    n_frames_output=10, num_objects=[2])
        dataset_valid = MovingMNIST(root=args.root, is_train=False, n_frames_input=10,
                                    n_frames_output=10, num_objects=[2])
    elif args.dataset == 'texibj':
        dataset_train = TexiBJ(root=args.root, is_train=True)
        dataset_valid = TexiBJ(root=args.root, is_train=False)
    elif args.dataset == 'sst':
        dataset_train = SST(root=args.root, is_train=True)
        dataset_valid = SST(root=args.root, is_train=False)
    else:
        dataset_train = Human(root=args.root, is_train=True)
        dataset_valid = Human(root=args.root, is_train=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train,
        num_replicas=args.world_size,
        rank=rank
    )
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers,
                                               pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4)

    dist.barrier()
    for epoch in range(start, nepochs):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003)
        
        # convert BN to SyncBatchNorm
        encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)

        for i, out in enumerate(train_loader, 0):
            input_tensor = out[1].cuda(gpu, non_blocking=True)
            target_tensor = out[2].cuda(gpu, non_blocking=True)
            # dist.barrier()
            loss = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer,
                                  criterion, teacher_forcing_ratio, args)
            loss_epoch += loss

            # if i == 4:
            #     break
        scheduler_mstep.step()
        train_losses.append(loss_epoch)
        if rank == 0:
            # record lr
            if policies != None:
                for j in range(len(policies)):
                    writer.add_scalar('Training/lr {}'.format(j+1),
                                      encoder_optimizer.param_groups[j]['lr'], epoch)

            else:
                writer.add_scalar('Training/lr 1', encoder_optimizer.param_groups[0]['lr'], epoch)
            # writer.add_scalar('Training/lr 2', encoder_optimizer.param_groups[1]['lr'], epoch)
            # writer.add_scalar('Training/lr 3', encoder_optimizer.param_groups[2]['lr'], epoch)
        if (epoch+1) % print_every == 0 and rank == 0:
            print('epoch ',epoch,  ' loss ',loss_epoch, ' time epoch ',time.time()-t0)
            writer.add_scalar('Training/epoch loss', loss_epoch, epoch)

        if (epoch+1) % eval_every == 0 :
            mse, mae,ssim, psnr, predictions = evaluate(encoder,test_loader, args)  # predictions, [10, 1, 64, 64]

            if rank == 0:
                writer.add_scalar('Validation/mse', mse, epoch)
                writer.add_scalar('Validation/ssim', ssim, epoch)
                writer.add_scalar('Validation/psnr', psnr, epoch)
                writer.add_image('Valid/pred 1', predictions[0], epoch, dataformats='CHW')
                writer.add_image('Valid/pred 2', predictions[4], epoch, dataformats='CHW')
                writer.add_image('Valid/pred 3', predictions[9], epoch, dataformats='CHW')
            # if mse > best_mse:
            #     print('new best at ', epoch)
            #     best_mse = mse
            # scheduler_enc.step(mse)
            scheduler_rop.step(mse)


            if torch.cuda.device_count() > 1:
                save_dict = {
                    'phydnet': encoder.module.state_dict(),
                    'optimizer': encoder_optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(save_dict, 'save/encoder_{}_{}.pth'.format(name, epoch))
            else:
                save_dict = {
                    'phydnet': encoder.state_dict(),
                    'optimizer': encoder_optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(save_dict,'save/encoder_{}_{}.pth'.format(name, epoch))

    writer.flush()
    writer.close()
    return train_losses


def evaluate(encoder, loader, args):
    '''

    :param encoder: model
    :param loader: test dataloader
    :return:
    '''

    total_mse, total_mae,total_ssim,total_bce = 0,0,0,0
    total_psnr = 0
    t0 = time.time()
    encoder.eval()
    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            input_tensor = out[1].cuda()    # [batch, 10, 1, 64, 64]
            target_tensor = out[2].cuda()
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]
            hidden1 = None
            hidden2 = None
            ex_input = None

            for ei in range(input_length-1):
                # encoder_output, encoder_hidden, _,hidden1,hidden2  = encoder(input_tensor[:,ei,:,:,:], hidden1,
                #                                                              hidden2, (ei==0))
                encoder_output, ex_input, _, hidden1, hidden2 = encoder(input=input_tensor[:, ei, :, :, :],
                                                                        ex_input_conv=ex_input,
                                                                        hidden1=hidden1,
                                                                        hidden2=hidden2,
                                                                        timestep=ei)

            decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                # decoder_output, decoder_hidden, output_image,hidden1,hidden2 = encoder(decoder_input, hidden1,
                #                                                                        hidden2, False, False)
                decoder_output, ex_input, output_image, hidden1, hidden2 = encoder(input=decoder_input,
                                                                        ex_input_conv=ex_input,
                                                                        hidden1=hidden1,
                                                                        hidden2=hidden2,
                                                                        timestep=di + input_length)
                decoder_input = output_image
                predictions.append(output_image.cpu().to(torch.float32))

            # input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions =  np.stack(predictions) # (10, batch_size, 1, 64, 64)
            predictions = predictions.swapaxes(0,1)  # (batch_size,10, 1, 64, 64)

            # our metric
            # mse_batch = np.mean((predictions - target) ** 2)
            # mae_batch = np.mean(np.abs(predictions - target))
            mse_batch = np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis=(0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch
            
            for a in range(0,target.shape[0]):
                for b in range(0,target.shape[1]):
                    total_ssim += ssim(target[a,b,0,], predictions[a,b,0,]) / (target.shape[0]*target.shape[1]) 
                    total_psnr += psnr(target[a,b,0,], predictions[a,b,0,]) / (target.shape[0]*target.shape[1])
            
            cross_entropy = -target*np.log(predictions) - (1-target) * np.log(1-predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (args.batch_size*target_length)
            total_bce +=  cross_entropy

            if i % 10 == 0:
                print(i)
            # if i == 20:
            #     break

     
    print('eval mse ', total_mse/len(loader), ' eval mae ', total_mae/len(loader))
    print('eval psnr ', total_psnr / len(loader), ' eval ssim ',total_ssim/len(loader))
    print(' time= ', time.time()-t0)
    return total_mse/len(loader),  total_mae/len(loader), total_ssim/len(loader), total_psnr/len(loader), predictions[0]


def visualize(encoder, loader, dataset, save_dir='results/'):
    '''
    save outputs of the network and the ground truth
    :param encoder: model
    :param loader: test dataset
    :param save_dir: the directory to save images
    the structure of save_dir:
    -- results
        -- pred
        -- gt
    :return:
    '''

    encoder.eval()

    pred_dir = os.path.join(save_dir, 'pred')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    gt_dir = os.path.join(save_dir, 'gt')
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)

    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            input_tensor = out[1].cuda()  # [batch, 10, 1, 64, 64]
            target_tensor = out[2].cuda()
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]
            hidden1 = None
            hidden2 = None
            ex_input = None

            for ei in range(input_length - 1):
                # encoder_output, encoder_hidden, _, hidden1, hidden2 = encoder(input_tensor[:, ei, :, :, :],
                #                                                               hidden1=hidden1, hidden2=hidden2,
                #                                                               first_timestep=(ei == 0))
                encoder_output, ex_input, _, hidden1, hidden2 = encoder(input=input_tensor[:, ei, :, :, :],
                                                                        ex_input_conv=ex_input,
                                                                        hidden1=hidden1,
                                                                        hidden2=hidden2,
                                                                        timestep=ei)

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                # decoder_output, decoder_hidden, output_image, hidden1, hidden2 = encoder(decoder_input, hidden1,
                #                                                                          hidden2, False, False)
                decoder_output, ex_input, output_image, hidden1, hidden2 = encoder(input=decoder_input,
                                                                        ex_input_conv=ex_input,
                                                                        hidden1=hidden1,
                                                                        hidden2=hidden2,
                                                                        timestep=di + input_length)
                decoder_input = output_image    # batch x 1x64x64  (2x32x32)
                predictions.append(torch.squeeze(output_image).cpu().numpy())

            input = torch.squeeze(input_tensor.cpu()).numpy()   # 10x64x64 （10x2x32x32)
            target = torch.squeeze(target_tensor.cpu()).numpy()

            if dataset == 'texibj':
                # for texibj
                shape = list(input.shape)
                ones = np.ones([shape[0], 1] + shape[2:], dtype=input.dtype)
                ones = ones * 0.514

                input[:, 0, :, :] = 0.88 * input[:, 0, :, :] + 0.01
                input[:, 1, :, :] = 0.87 * input[:, 1, :, :] + 0.05
                input = np.concatenate([ones, input], axis=1)    # 10x3x32x32
                input = np.transpose(input, (0, 2, 3, 1))

                target[:, 0, :, :] = 0.88 * target[:, 0, :, :] + 0.01
                target[:, 1, :, :] = 0.87 * target[:, 1, :, :] + 0.05
                target = np.concatenate([ones, target], axis=1)
                target = np.transpose(target, (0, 2, 3, 1))

                # predictions = [np.transpose(np.concatenate([ones[0], x], axis=0), (1, 2, 0)) for x in predictions]
                # new_predictions = []
                for x in predictions:
                    x[0] = 0.88 * x[0] + 0.01
                    x[1] = 0.87 * x[1] + 0.05
                    x = np.concatenate([ones[0], x], axis=0)
                    x = np.transpose(x, (1, 2, 0))
                    # new_predictions.append(x)

            # set up directories
            if not os.path.exists(os.path.join(pred_dir, str(i))):
                os.mkdir(os.path.join(pred_dir, str(i)))
            if not os.path.exists(os.path.join(gt_dir, str(i))):
                os.mkdir(os.path.join(gt_dir, str(i)))

            # gt
            for k in range(0, 20):

                if k < 10:
                    image = np.transpose(input[k], (1, 2, 0))
                    cv.imwrite(os.path.join(gt_dir, str(i), '{}.jpg'.format(k)), image * 255.0)
                else:
                    image = np.transpose(target[k-10], (1, 2, 0))
                    cv.imwrite(os.path.join(gt_dir, str(i), '{}.jpg'.format(k)), image * 255.0)

            # prediction
            for k in range(0, 10):
                image = np.transpose(predictions[k], (1, 2, 0))
                cv.imwrite(os.path.join(pred_dir, str(i), '{}.jpg'.format(k)), image * 255.0)

            #     break



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
   
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/', help='directory of dataset')
    parser.add_argument('--model', type=str, default=None, help='checkpoint of last training')
    parser.add_argument('--pre_model', type=str, default=None, help='checkpoint of pretrained model')
    parser.add_argument('--r2d_model', type=str, default='./save/r2plus1d_layer.pth',
                        help='checkpoint of r(2+1)d model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--nepochs', type=int, default=2001, help='nb of epochs')
    parser.add_argument('--print_every', type=int, default=1, help='')
    parser.add_argument('--eval_every', type=int, default=5, help='')
    parser.add_argument('--save_name', type=str, default='convlstm', help='')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--eval', action='store_true', help='calculate metrics')
    parser.add_argument('--visual', action='store_true', help='output images')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--num_workers' ,default=4, type=int)
    parser.add_argument('--dataset', type=str, default='mm',  help='mm, texibj, sst, or human')

    args = parser.parse_args()


    
    # train(args)
    test(args)
