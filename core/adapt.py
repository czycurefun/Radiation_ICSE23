
import os
import torch
import torch.optim as optim
from torch import nn
from params import param
from utils import make_cuda
from sklearn.decomposition import PCA


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):


    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) 
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) 
  
def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    target_size = int(target.size()[0])
    if batch_size == target_size:
        kernels = guassian_kernel(source, target,
                                kernel_mul=kernel_mul, 	
                                    kernel_num=kernel_num, 	
                                fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size] # Source<->Source
        YY = kernels[batch_size:, batch_size:] # Target<->Target
        XY = kernels[:batch_size, batch_size:] # Source<->Target
        YX = kernels[batch_size:, :batch_size] # Target<->Source
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    elif batch_size < target_size:
        kernels = guassian_kernel(source, target,
                                kernel_mul=kernel_mul, 	
                                    kernel_num=kernel_num, 	
                                fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size] # Source<->Source
        YY = kernels[batch_size:batch_size*2, batch_size:batch_size*2] # Target<->Target
        XY = kernels[:batch_size, batch_size:batch_size*2] # Source<->Target
        YX = kernels[batch_size:batch_size*2, :batch_size] # Target<->Source
        loss = torch.mean(XX + YY - XY -YX) 
        return loss
    else:
        kernels = guassian_kernel(source, target,
                                kernel_mul=kernel_mul, 	
                                    kernel_num=kernel_num, 	
                                fix_sigma=fix_sigma)
        XX = kernels[:target_size, :target_size] # Source<->Source
        YY = kernels[target_size:target_size*2, target_size:target_size*2] # Target<->Target
        XY = kernels[:target_size, target_size:target_size*2] # Source<->Target
        YX = kernels[target_size:target_size*2, :target_size] # Target<->Source
        loss = torch.mean(XX + YY - XY -YX)
                                                                   
        return loss

def train_tgt(args, src_encoder, src_encoder2, src_encoder3,src_encoder4, src_encoder5, src_encoder6,
              tgt_encoder, critic,
              src_data_loader, src_data_loader2, src_data_loader3,src_data_loader4, src_data_loader5, src_data_loader6, tgt_data_loader):


    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=param.c_learning_rate,
                               betas=(param.beta1, param.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=param.d_learning_rate,
                                  betas=(param.beta1, param.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(args.num_epochs):

        data_zip = enumerate(zip(src_data_loader, src_data_loader2, src_data_loader3, src_data_loader4, src_data_loader5, src_data_loader6, tgt_data_loader))
        for step, ((input_ids_src, attention_mask_src, token_type_ids_src, labels_src), (input_ids_src2, attention_mask_src2, token_type_ids_src2, labels_src2),
                   (input_ids_src3, attention_mask_src3, token_type_ids_src3, labels_src3), (input_ids_src4, attention_mask_src4, token_type_ids_src4, labels_src4),
                   (input_ids_src5, attention_mask_src5, token_type_ids_src5, labels_src5), (input_ids_src6, attention_mask_src6, token_type_ids_src6, labels_src6),
                   (input_ids_tgt, attention_mask_tgt, token_type_ids_tgt, labels_tgt)) in data_zip:

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            
            _, feat_src = src_encoder(input_ids_src, attention_mask_src, token_type_ids_src)
            # _, feat_src2 = src_encoder(input_ids_src2, attention_mask_src2, token_type_ids_src2)
            # _, feat_src3 = src_encoder(input_ids_src3, attention_mask_src3, token_type_ids_src3)
            # _, feat_src4 = src_encoder(input_ids_src4, attention_mask_src4, token_type_ids_src4)
            # _, feat_src5 = src_encoder5(input_ids_src5, attention_mask_src5, token_type_ids_src5)
            # _, feat_src6 = src_encoder6(input_ids_src6, attention_mask_src6, token_type_ids_src6)
            _, feat_tgt = tgt_encoder(input_ids_tgt, attention_mask_tgt, token_type_ids_tgt)
            pca = PCA(n_components=2)
            # distance_loss = mmd(feat_src, feat_tgt) 

            # feat_src = torch.from_numpy(pca.fit_transform(feat_src.cpu())).to(device='cuda').float()  
            # feat_src2 = torch.from_numpy(pca.fit_transform(feat_src2.cpu())).to(device='cuda').float() 
            # feat_src3 = torch.from_numpy(pca.fit_transform(feat_src3.cpu())).to(device='cuda').float() 
            # feat_tgt = torch.from_numpy(pca.fit_transform(feat_tgt.cpu())).to(device='cuda').float() 
            # feat_concat = torch.cat((torch.from_numpy(feat_src), torch.from_numpy(feat_src2), torch.from_numpy(feat_src3), torch.from_numpy(feat_tgt)), 0)
            # feat_concat = feat_concat
            # feat_concat = torch.cat((feat_src, feat_src2, feat_src3, feat_src4,  feat_tgt), 0)
            feat_concat = torch.cat((feat_src,   feat_tgt), 0)
            # predict on discriminator
            pred_concat = critic(feat_concat.detach())
            aa = torch.rand(32)

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src.size(0)).long())
            # label_src2 = make_cuda(torch.ones(feat_src2.size(0)).long())
            # label_src3 = make_cuda(torch.ones(feat_src3.size(0)).long())
            # label_src4 = make_cuda(torch.ones(feat_src4.size(0)).long())
            # label_src5 = make_cuda(torch.full_like(torch.rand(feat_src5.size(0)),5).long())
            # label_src6 = make_cuda(torch.full_like(torch.rand(feat_src6.size(0)),6).long())
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0)).long())
            # label_concat = torch.cat((label_src, label_src2, label_src3, label_src4, label_tgt), 0)
            label_concat = torch.cat((label_src, label_tgt), 0)
            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_tgt.zero_grad()

            # extract and target features
            _, feat_src = src_encoder(input_ids_src, attention_mask_src, token_type_ids_src)
            _, feat_tgt = tgt_encoder(input_ids_tgt, attention_mask_tgt, token_type_ids_tgt)
            # feat_tgt = pca.fit_transform(feat_tgt.cpu())
            # feat_tgt = torch.from_numpy(feat_tgt)
            # feat_tgt = feat_tgt.to(device='cuda').float() 
            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            
            label_tgt = make_cuda(torch.ones(int(feat_tgt.size(0))).long())
            # label_tgt2 = make_cuda(torch.full_like(torch.rand(int(feat_tgt.size(0)/6)),2).long())
            # label_tgt3 = make_cuda(torch.full_like(torch.rand(int(feat_tgt.size(0)/6)),3).long())
            # label_tgt4 = make_cuda(torch.full_like(torch.rand(int(feat_tgt.size(0)/6)),4).long())
            # label_tgt5 = make_cuda(torch.full_like(torch.rand(int(feat_tgt.size(0)/6)),5).long())
            # label_tgt6 = make_cuda(torch.full_like(torch.rand(feat_tgt.size(0) - int(feat_tgt.size(0)/6)*5),6).long())
            # label_tgt3 = make_cuda(torch.full_like(torch.rand(feat_tgt.size(0) - int(feat_tgt.size(0)/3)*2),3).long())
            # label_tgtconcat = torch.cat((label_tgt, label_tgt2, label_tgt3, label_tgt4, label_tgt5, label_tgt6), 0)

            # compute loss for target encoder
            distance_loss = mmd(feat_src, feat_tgt) 
            loss_tgt = criterion(pred_tgt, label_tgt)
            Loss_merge = loss_tgt + distance_loss
            Loss_merge.backward()

            # optimize target encoder
            optimizer_tgt.step()

            if (step + 1) % args.log_step == 0:
                print("Epoch [%.3d/%.3d] Step [%.2d/%.2d]:"
                      "d_loss=%.4f g_loss=%.4f acc=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         loss_critic.item(),
                         loss_tgt.item(),
                         acc.item()))

        if (epoch + 1) % args.save_step == 0:
            torch.save(critic.state_dict(), os.path.join(
                args.model_root,
                "RL-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                args.model_root,
                "RL-target-encoder-{}.pt".format(epoch + 1)))

    return tgt_encoder
