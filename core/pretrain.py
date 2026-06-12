
import torch.nn as nn
import torch.optim as optim
from params import param
from utils import save_model
from transformers import  get_linear_schedule_with_warmup
from transformers import AdamW
import torch

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

#三个src
def train_src(args, encoder, encoder2, encoder3, encoder4, encoder5, encoder6,
              classifier, data_loader, data_loader_eval, data_loader2, data_loader_eval2, data_loader3, data_loader_eval3,
               data_loader4, data_loader_eval4, data_loader5, data_loader_eval5, data_loader6, data_loader_eval6, tgt_data_loader):
    earlystop = EarlyStop(args.patience)


    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()) ,
        lr=param.c_learning_rate,
        betas=(param.beta1, param.beta2))
    # optimizer = AdamW(encoder.parameters(), lr=param.c_learning_rate, eps=args.adam_epsilon)
    #新加的学习率
    t_total = len(data_loader) * args.num_epochs_pre
    warmup_steps = int(0.1 * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    # encoder2.train()
    # encoder3.train()
    # encoder4.train()
    # encoder5.train()
    # encoder6.train()


    for epoch in range(args.num_epochs_pre):
        #2个src
        # data_zip = enumerate(zip(data_loader, data_loader2))
        #3个src
        data_zip = enumerate(zip(data_loader, data_loader2, data_loader3, data_loader4, data_loader5, data_loader6, tgt_data_loader))
        # for step, (reviews, labels) in enumerate(data_loader):
        # for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_loader):
        for step, ((input_ids, attention_mask, token_type_ids, labels),(input_ids2, attention_mask2, token_type_ids2, labels2),
                   (input_ids3, attention_mask3, token_type_ids3, labels3),(input_ids4, attention_mask4, token_type_ids4, labels4),
                   (input_ids5, attention_mask5, token_type_ids5, labels5),(input_ids6, attention_mask6, token_type_ids6, labels6),
                   (input_ids_tgt, attention_mask_tgt, token_type_ids_tgt, labels_tgt)) in data_zip:
            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            # preds = classifier(encoder(reviews))
            CLS1, CNN1  = encoder(input_ids, attention_mask, token_type_ids)
            CLS_tgt, CNN1  = encoder(input_ids_tgt, attention_mask_tgt, token_type_ids_tgt)
            # CLS2, CNN2  = encoder(input_ids2, attention_mask2, token_type_ids2)
            # CLS3, CNN3  = encoder(input_ids3, attention_mask3, token_type_ids3)
            # CLS4, CNN4  = encoder(input_ids4, attention_mask4, token_type_ids4)
            # CLS5, CNN5  = encoder5(input_ids5, attention_mask5, token_type_ids5)
            # CLS6, CNN6  = encoder6(input_ids6, attention_mask6, token_type_ids6)
            
            preds = classifier(CLS1)
            # preds2 = classifier(CLS2)
            # preds3 = classifier(CLS3)
            # preds4 = classifier(CLS4)
            # preds5 = classifier(CLS5)
            # preds6 = classifier(CLS6)
            # preds_concat = torch.cat((preds, preds2, preds3, preds4), 0)
            # label_concat = torch.cat((labels, labels2, labels3, labels4), 0)
            loss = criterion(preds, labels)

            distance_loss = mmd(CLS1, CLS_tgt) 

            # loss = criterion(preds_concat, label_concat)
            Loss_merge = distance_loss + loss
            

            # optimize source classifier
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print step info
            if (step + 1) % args.log_step_pre == 0:
                print("Epoch [%.3d/%.3d] Step [%.2d/%.2d]: loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs_pre,
                         step + 1,
                         len(data_loader),
                         loss.item()))

        # eval model on test set
        if (epoch + 1) % args.eval_step_pre == 0:
            eval_src(encoder, encoder2, encoder3, encoder4, encoder5, encoder6, classifier, data_loader, data_loader2, data_loader3, data_loader4, data_loader5, data_loader6)
            earlystop.update(eval_src(encoder, encoder2, encoder3, encoder4, encoder5, encoder6, classifier, data_loader_eval, data_loader_eval2, data_loader_eval3 , data_loader_eval4, data_loader_eval5, data_loader_eval6))
            print()

        # save model parameters
        if (epoch + 1) % args.save_step_pre == 0:
            # pass
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

        if earlystop.stop:
            break

    # # save final model
    save_model(encoder, "RL-source-encoder-final.pt")
    save_model(classifier, "RL-source-classifier-final.pt")

    return encoder, encoder2, encoder3, encoder4, encoder5, encoder6, classifier

def eval_src(encoder, encoder2, encoder3, encoder4, encoder5, encoder6,
             classifier, data_loader, data_loader2,data_loader3, data_loader4, data_loader5,data_loader6):


    # init loss and accuracy
    loss = 0
    acc = 0
    TP = 0
    FP = 0
    FN = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    # for (reviews, labels) in data_loader:
    data_zip = enumerate(zip(data_loader, data_loader2, data_loader3, data_loader4, data_loader5, data_loader6))
    # for (input_ids, attention_mask, token_type_ids, labels) in data_loader:
    for step, ((input_ids, attention_mask, token_type_ids, labels),(input_ids2, attention_mask2, token_type_ids2, labels2),
               (input_ids3, attention_mask3, token_type_ids3, labels3),(input_ids4, attention_mask4, token_type_ids4, labels4),
               (input_ids5, attention_mask5, token_type_ids5, labels5),(input_ids6, attention_mask6, token_type_ids6, labels6)) in data_zip:

        # preds = classifier(encoder(reviews))
        CLS1, CNN1  = encoder(input_ids, attention_mask, token_type_ids)
        # CLS2, CNN2  = encoder(input_ids2, attention_mask2, token_type_ids2)
        # CLS3, CNN3  = encoder(input_ids3, attention_mask3, token_type_ids3)
        # CLS4, CNN4  = encoder(input_ids4, attention_mask4, token_type_ids4)
        # CLS5, CNN5  = encoder5(input_ids5, attention_mask5, token_type_ids5)
        # CLS6, CNN6  = encoder6(input_ids6, attention_mask6, token_type_ids6)
        
        preds = classifier(CLS1)
        # preds2 = classifier(CLS2)
        # preds3 = classifier(CLS3)
        # preds4 = classifier(CLS4)
        # preds5 = classifier(CLS5)
        # preds6 = classifier(CLS6)
        # preds_concat = torch.cat((preds, preds2, preds3, preds4), 0)
        # label_concat = torch.cat((labels, labels2, labels3, labels4), 0)
        loss += criterion(preds, labels).item()
        # loss += criterion(preds_concat, label_concat).item()
        # pred_cls = preds.data.max(1)[1]
        pred_cls = preds.data.max(1)[1]
        TP += ((pred_cls == 1) & (labels == 1)).sum()
        FP += ((pred_cls == 1) & (labels == 0)).sum()
        FN += ((pred_cls == 0) & (labels == 1)).sum()
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f, precision = %.4f, recall = %.4f, F1 = %.4f" % (loss, acc, precision,recall,F1))

    return loss


class EarlyStop:
    def __init__(self, patience):
        self.count = 0
        self.maxAcc = 0
        self.patience = patience
        self.stop = False

    def update(self, acc):
        if acc < self.maxAcc:
            self.count += 1
        else:
            self.count = 0
            self.maxAcc = acc

        if self.count > self.patience:
            self.stop = True
