

import torch.nn as nn
import torch


def eval_tgt(encoder, classifier, data_loader, rawone, rawtwo):

    # init loss and accuracy
    loss = 0
    acc = 0
    TP = 0
    FP = 0
    FN = 0
    # set loss function
    batch_num = 0
    criterion = nn.CrossEntropyLoss()
    log_path = "result/alltoeasyquarter/result.txt"
    for (input_ids, attention_mask, token_type_ids, labels) in data_loader:
        # preds = classifier(encoder(reviews))
        CLS, CNN = encoder(input_ids, attention_mask, token_type_ids)
        preds = classifier(CLS)
        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        TP += ((pred_cls == 1) & (labels == 1)).sum()
        FP += ((pred_cls == 1) & (labels == 0)).sum()
        FN += ((pred_cls == 0) & (labels == 1)).sum()
        acc += pred_cls.eq(labels.data).cpu().sum().item()
        with open(log_path, "a",encoding="utf-8") as fout:
            for num , each in enumerate(pred_cls):
                fout.write("********************************\n")
                fout.write("query1:\n")
                fout.write(rawone[num + batch_num*16])
                fout.write("\n")
                fout.write("query2:\n") 
                fout.write(rawtwo[num + batch_num*16])
                fout.write("\n")
                fout.write("groudtruth:\n") 
                if labels[num] == 1:
                    fout.write("1")
                elif labels[num] == 0:
                    fout.write("0")
                fout.write("\n")
                fout.write("predict:\n") 
                if pred_cls[num] == 1:
                    fout.write("1")
                elif pred_cls[num] == 0:
                    fout.write("0")
                fout.write("\n")
        batch_num += 1

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f, precision = %.4f, recall = %.4f, F1 = %.4f" % (loss, acc, precision,recall,F1))
