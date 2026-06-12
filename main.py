

import torch
from params import param
from core import eval_src, eval_tgt, train_src, train_tgt
from models import BERTEncoder, BERTClassifier, Discriminator
from utils import read_data, get_data_loader, init_model, init_random_seed,readDataFromFile
from transformers import BertTokenizer
import argparse
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer  
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
import math
if __name__ == '__main__':
    print("ok")
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="easy", choices=["easy", "Infusion", "CM1", "EBT"],
                        help="Specify src dataset")
    
    parser.add_argument('--src2', type=str, default="Infusion", choices=["easy", "Infusion", "CM1", "EBT"],
                        help="Specify src dataset2")
    
    parser.add_argument('--src3', type=str, default="CM1", choices=["easy", "Infusion", "CM1", "EBT"],
                        help="Specify src dataset2")
    
    parser.add_argument('--src4', type=str, default="EBT", choices=["easy", "Infusion", "CM1", "EBT"],
                        help="Specify src dataset2")
    
    parser.add_argument('--src5', type=str, default="CCHIT", choices=["easy", "Infusion", "CM1", "EBT"],
                        help="Specify src dataset2")
    
    parser.add_argument('--src6', type=str, default="EBT", choices=["easy", "Infusion", "CM1", "EBT"],
                        help="Specify src dataset2")

    parser.add_argument('--tgt', type=str, default="EBT", choices=["easy", "Infusion", "CM1", "EBT"],
                        help="Specify tgt dataset")

    parser.add_argument('--enc_train', default=False, action='store_true',
                        help='Train source encoder')

    parser.add_argument('--seqlen', type=int, default=256,
                        help="Specify maximum sequence length")

    parser.add_argument('--patience', type=int, default=5,
                        help="Specify patience of early stopping for pretrain")
    parser.add_argument('--num_epochs_pre', type=int, default=50,
                        help="Specify the number of epochs for pretrain")

    parser.add_argument('--log_step_pre', type=int, default=1,
                        help="Specify log step size for pretrain")

    parser.add_argument('--eval_step_pre', type=int, default=10,
                        help="Specify eval step size for pretrain")

    parser.add_argument('--save_step_pre', type=int, default=100,
                        help="Specify save step size for pretrain")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")

    parser.add_argument('--save_step', type=int, default=100,
                        help="Specify save step size for adaptation")
    
    parser.add_argument('--model_root', type=str, default="result/alltoeasyquarter",
                        help="save model path")
    
    parser.add_argument('--adam_epsilon', type=float, default="1e-8",
                        help="vv")
    

    args = parser.parse_args()

    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("enc_train: " + str(args.enc_train))
    print("seqlen: " + str(args.seqlen))
    print("patience: " + str(args.patience))
    print("num_epochs_pre: " + str(args.num_epochs_pre))
    print("log_step_pre: " + str(args.log_step_pre))
    print("eval_step_pre: " + str(args.eval_step_pre))
    print("save_step_pre: " + str(args.save_step_pre))
    print("num_epochs: " + str(args.num_epochs))
    print("log_step: " + str(args.log_step))
    print("save_step: " + str(args.save_step))

    # init random seed
    init_random_seed(param.manual_seed)

    # preprocess data
    print("=== Processing datasets ===")
    
    src_train = readDataFromFile('./data/processed/' + args.src + '/newtrain.txt')
    src_test = readDataFromFile('./data/processed/' + args.src + '/newtest.txt')
    tgt_train = readDataFromFile('./data/processed/' + args.tgt + '/newtrain.txt')
    tgt_test = readDataFromFile('./data/processed/' + args.tgt + '/newtest.txt')
    
    src_train2 = readDataFromFile('./data/processed/' + args.src2 + '/newtrain.txt')
    src_test2 = readDataFromFile('./data/processed/' + args.src2 + '/newtest.txt')
    
    src_train3 = readDataFromFile('./data/processed/' + args.src3 + '/newtrain.txt')
    src_test3 = readDataFromFile('./data/processed/' + args.src3 + '/newtest.txt')
    
    src_train4 = readDataFromFile('./data/processed/' + args.src4 + '/newtrain.txt')
    src_test4 = readDataFromFile('./data/processed/' + args.src4 + '/newtest.txt')
    
    src_train5 = readDataFromFile('./data/processed/' + args.src5 + '/newtrain.txt')
    src_test5 = readDataFromFile('./data/processed/' + args.src5 + '/newtest.txt')
    
    src_train6 = readDataFromFile('./data/processed/' + args.src6 + '/newtrain.txt')
    src_test6 = readDataFromFile('./data/processed/' + args.src6 + '/newtest.txt')

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased',do_lower_case=True)

    src_train_sequences = []
    src_test_sequences = []
    tgt_train_sequences = []
    tgt_test_sequences = []
    
    src_train_labels = []
    src_test_labels  = []
    tgt_train_labels  = []
    tgt_test_labels  = []
    
    tgt_train_oneraw = []
    tgt_train_tworaw = []
    tgt_test_oneraw = []
    tgt_test_tworaw = []
    
    #新增第二个源数据
    src_train_sequences2 = []
    src_test_sequences2 = []
    
    src_train_labels2 = []
    src_test_labels2  = []
    
    src_train_sequences3 = []
    src_test_sequences3 = []
    
    src_train_labels3 = []
    src_test_labels3  = []
    
    src_train_sequences4 = []
    src_test_sequences4 = []
    
    src_train_labels4 = []
    src_test_labels4  = []
    
    src_train_sequences5 = []
    src_test_sequences5 = []
    
    src_train_labels5 = []
    src_test_labels5  = []
    
    src_train_sequences6 = []
    src_test_sequences6 = []
    
    src_train_labels6 = []
    src_test_labels6  = []
    
    src_train_labels_keyword = []
    src_train_sequences_keyword = []
    
    src_test_labels_keyword = []
    src_test_sequences_keyword = []
    
    tgt_train_labels_keyword = []
    tgt_train_sequences_keyword = []
    
    tgt_test_labels_keyword = []
    tgt_test_sequences_keyword = []

    
    for snum,each in enumerate(src_train):
     
        sentenceone = each[0]
        sentencetwo = each[1]

        constraint_num = 0
        indexed_tokens = tokenizer.encode_plus(text=sentenceone,
                                                    text_pair=sentencetwo,
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids = indexed_tokens['input_ids']
        src_train_labels.append(int(each[2]))
        src_train_sequences.append(indexed_tokens)
        
       
    for snum,each in enumerate(src_test):
        sentenceone = each[0]
        sentencetwo = each[1]
        indexed_tokens = tokenizer.encode_plus(text=sentenceone,
                                                    text_pair=sentencetwo,
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_test_labels.append(int(each[2]))
        src_test_sequences.append(indexed_tokens)
        

    sentence_list = []
    for sentence in tgt_train:
        sentence_list.append(remove_stopwords(sentence[0]) + remove_stopwords(sentence[1]))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentence_list[:10000]) 
    words = vectorizer.get_feature_names()
    X_mat=X.toarray()
    doc_num=X.shape[0]
    idf_list = []
    for index in range(len(words)):
        idf=math.log(doc_num/(sum(np.sign(X_mat[:,index]))+1))
        idf_list.append(idf)
        # print(words[index]+' '+str(idf))
    emerge_list = zip(idf_list, words)
    sort_list = sorted(emerge_list)
    idf_words = []
    extract_num = 0
    for each in sort_list:
        if extract_num < 19:
            if each[1] != 'the' and  each[1] != 'shall' and each[1].isdigit() == False:
                idf_words.append(each[1])
            extract_num += 1
        else:
            break
    for snum,each in enumerate(tgt_train):    
        sentenceone = each[0]
        sentencetwo = each[1]
        for eachword in idf_words:
            if " " + eachword.lower() + " " in each[0]:
                sentenceone = each[0].replace(  " " + eachword.lower() + " " ," [MASK] ")
                each[0] = sentenceone
            # if  eachword.lower()  in each[0]:
            #     sentenceone = each[0].replace(  eachword.upper() ,"[MASK]")
            #     each[0] = sentenceone
            # if "Steps" in each[0]:
            #     sentenceone = each[0].replace("Steps","[MASK]")
            #     each[0] = sentenceone
            # if "Preconditions" in each[0]:
            #     sentenceone = each[0].replace("Preconditions","[MASK]")
            #     each[0] = sentenceone
            #INFUSION
            # if "Subsystem" in each[0]:
            #     sentenceone = each[0].replace("Subsystem","[MASK]")
            #     each[0] = sentenceone
            # if "Pump" in each[0]:
            #     sentenceone = each[0].replace("Pump","[MASK]")
            #     each[0] = sentenceone
            # if eachword.upper() in each[0]:
            #     sentenceone = each[0].replace(eachword.upper(),"[MASK]")
            #     each[0] = sentenceone
            if   " " + eachword.lower() + " "  in each[1]:
                sentencetwo = each[1].replace( " " + eachword.lower() + " " ," [MASK] ")
                each[1] = sentencetwo
            # if  eachword.upper()  in each[1]:
            #     sentencetwo = each[1].replace(  eachword.upper() ,"[MASK]")
            #     each[1] = sentencetwo
            # if "Steps" in each[1]:
            #     sentencetwo = each[1].replace("Steps","[MASK]")
            #     each[1] = sentencetwo
            # if "Preconditions" in each[1]:
            #     sentencetwo = each[1].replace("Preconditions","[MASK]")
            #     each[1] = sentencetwo
            # if "Subsystem" in each[1]:
            #     sentencetwo = each[1].replace("Subsystem","[MASK]")
            #     each[1] = sentencetwo
            # if "Pump" in each[1]:
            #     sentencetwo = each[1].replace("Pump","[MASK]")
            #     each[1] = sentencetwo
            #Infusion

            # if eachword.upper() in each[1]:
            #     sentencetwo = each[1].replace(eachword.upper(),"[MASK]")
            #     each[1] = sentencetwo
        tgt_train_oneraw.append(sentenceone)
        tgt_train_tworaw.append(sentencetwo)
        indexed_tokens = tokenizer.encode_plus(text=sentenceone,
                                                    text_pair=sentencetwo,
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        tgt_train_labels.append(int(each[2]))
        tgt_train_sequences.append(indexed_tokens)
        

    sentence_list = []
    for sentence in tgt_test:
        sentence_list.append(remove_stopwords(sentence[0]) + remove_stopwords(sentence[1]))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentence_list[:10000]) 
    words = vectorizer.get_feature_names()
    X_mat=X.toarray()
    doc_num=X.shape[0]
    idf_list = []
    for index in range(len(words)):
        idf=math.log(doc_num/(sum(np.sign(X_mat[:,index]))+1))
        idf_list.append(idf)
        # print(words[index]+' '+str(idf))
    emerge_list = zip(idf_list, words)
    sort_list = sorted(emerge_list)
    idf_words = []
    extract_num = 0
    for each in sort_list:
        if extract_num < 19:
            if each[1] != 'the' and  each[1] != 'shall' and each[1].isdigit() == False:
                idf_words.append(each[1])
            extract_num += 1
        else:
            break
    for snum,each in enumerate(tgt_test):

        sentenceone = each[0]
        sentencetwo = each[1]
        constraint_num = 0
        for eachword in idf_words:
            if " " + eachword.lower() + " " in each[0]:
                sentenceone = each[0].replace(  " " + eachword.lower() + " " ," [MASK] ")
                each[0] = sentenceone
            # if  eachword.upper()  in each[0]:
            #     sentenceone = each[0].replace(  eachword.upper() ,"[MASK]")
            #     each[0] = sentenceone
            # if "Steps" in each[0]:
            #     sentenceone = each[0].replace("Steps","[MASK]")
            #     each[0] = sentenceone
            # if "Preconditions" in each[0]:
            #     sentenceone = each[0].replace("Preconditions","[MASK]")
            #     each[0] = sentenceone
            #INFUSION
            # if "Subsystem" in each[0]:
            #     sentenceone = each[0].replace("Subsystem","[MASK]")
            #     each[0] = sentenceone
            # if "Pump" in each[0]:
            #     sentenceone = each[0].replace("Pump","[MASK]")
            #     each[0] = sentenceone
            # if eachword.upper() in each[0]:
            #     sentenceone = each[0].replace(eachword.upper(),"[MASK]")
            #     each[0] = sentenceone
            if   " " + eachword.lower() + " "  in each[1]:
                sentencetwo = each[1].replace( " " + eachword.lower() + " " ," [MASK] ")
                each[1] = sentencetwo
            # if   eachword.upper()   in each[1]:
            #     sentencetwo = each[1].replace(  eachword.upper() ,"[MASK]")
            #     each[1] = sentencetwo
            # if "Steps" in each[1]:
            #     sentencetwo = each[1].replace("Steps","[MASK]")
            #     each[1] = sentencetwo
            # if "Preconditions" in each[1]:
            #     sentencetwo = each[1].replace("Preconditions","[MASK]")
            #     each[1] = sentencetwo
            # if "Subsystem" in each[1]:
            #     sentencetwo = each[1].replace("Subsystem","[MASK]")
            #     each[1] = sentencetwo
            # if "Pump" in each[1]:
            #     sentencetwo = each[1].replace("Pump","[MASK]")
            #     each[1] = sentencetwo
            #Infusion
            # if eachword.upper() in each[1]:
            #     sentencetwo = each[1].replace(eachword.upper(),"[MASK]")
            #     each[1] = sentencetwo
        tgt_test_oneraw.append(sentenceone)
        tgt_test_tworaw.append(sentencetwo)
        indexed_tokens = tokenizer.encode_plus(text=sentenceone,
                                                    text_pair=sentencetwo,
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        tgt_test_labels.append(int(each[2]))
        tgt_test_sequences.append(indexed_tokens)
        

    for each in src_train2:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_train_labels2.append(int(each[2]))
        src_train_sequences2.append(indexed_tokens)
        
    for each in src_test2:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_test_labels2.append(int(each[2]))
        src_test_sequences2.append(indexed_tokens)

    for each in src_train3:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_train_labels3.append(int(each[2]))
        src_train_sequences3.append(indexed_tokens)
        
    for each in src_test3:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_test_labels3.append(int(each[2]))
        src_test_sequences3.append(indexed_tokens)
        
    for each in src_train4:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_train_labels4.append(int(each[2]))
        src_train_sequences4.append(indexed_tokens)
        
    for each in src_test4:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_test_labels4.append(int(each[2]))
        src_test_sequences4.append(indexed_tokens)
        
    for each in src_train5:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_train_labels5.append(int(each[2]))
        src_train_sequences5.append(indexed_tokens)
        
    for each in src_test5:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_test_labels5.append(int(each[2]))
        src_test_sequences5.append(indexed_tokens)
        
    for each in src_train6:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_train_labels6.append(int(each[2]))
        src_train_sequences6.append(indexed_tokens)
        
    for each in src_test6:
        indexed_tokens = tokenizer.encode_plus(text=each[0],
                                                    text_pair=each[1],
                                                    max_length=args.seqlen,
                                                    truncation=True,
                                                    padding='max_length')
        src_test_labels6.append(int(each[2]))
        src_test_sequences6.append(indexed_tokens)
    


    # load dataset
    src_data_loader = get_data_loader(src_train_sequences, src_train_labels, args.seqlen)
    src_data_loader_eval = get_data_loader(src_test_sequences, src_test_labels, args.seqlen)
    tgt_data_loader = get_data_loader(tgt_train_sequences, tgt_train_labels, args.seqlen)
    tgt_data_loader_eval = get_data_loader(tgt_test_sequences, tgt_test_labels, args.seqlen)
 
    src_data_loader2 = get_data_loader(src_train_sequences2, src_train_labels2, args.seqlen)
    src_data_loader_eval2 = get_data_loader(src_test_sequences2, src_test_labels2, args.seqlen)
    
    src_data_loader3 = get_data_loader(src_train_sequences3, src_train_labels3, args.seqlen)
    src_data_loader_eval3 = get_data_loader(src_test_sequences3, src_test_labels3, args.seqlen)
    
    src_data_loader4 = get_data_loader(src_train_sequences4, src_train_labels4, args.seqlen)
    src_data_loader_eval4 = get_data_loader(src_test_sequences4, src_test_labels4, args.seqlen)
    
    src_data_loader5 = get_data_loader(src_train_sequences5, src_train_labels5, args.seqlen)
    src_data_loader_eval5 = get_data_loader(src_test_sequences5, src_test_labels5, args.seqlen)
    
    src_data_loader6 = get_data_loader(src_train_sequences6, src_train_labels6, args.seqlen)
    src_data_loader_eval6 = get_data_loader(src_test_sequences6, src_test_labels6, args.seqlen)

    print("=== Datasets successfully loaded ===")

    # load models
    src_encoder = init_model(BERTEncoder(),
                             restore=param.src_encoder_restore)
    src_classifier = init_model(BERTClassifier(),
                                restore=param.src_classifier_restore)
    tgt_encoder = init_model(BERTEncoder(),
                             restore=param.tgt_encoder_restore)
    critic = init_model(Discriminator(),
                        restore=param.d_model_restore)

    src_encoder2 = init_model(BERTEncoder(),
                             restore=param.src_encoder_restore2)
    src_encoder3 = init_model(BERTEncoder(),
                             restore=param.src_encoder_restore3)
    src_encoder4 = init_model(BERTEncoder(),
                             restore=param.src_encoder_restore4)
    src_encoder5 = init_model(BERTEncoder(),
                             restore=param.src_encoder_restore5)
    src_encoder6 = init_model(BERTEncoder(),
                             restore=param.src_encoder_restore6)

    # freeze encoder params
    if not args.enc_train:
        for param in src_encoder.parameters():
            param.requires_grad = False
        for param in src_encoder2.parameters():
            param.requires_grad = False
        for param in src_encoder3.parameters():
            param.requires_grad = False
        for param in src_encoder4.parameters():
            param.requires_grad = False
        for param in src_encoder5.parameters():
            param.requires_grad = False
        for param in src_encoder6.parameters():
            param.requires_grad = False

    # train source model
    print("=== Training classifier for source domain ===")

    

    src_encoder, src_encoder2, src_encoder3, src_encoder4, src_encoder5, src_encoder6, src_classifier = train_src(
        args, src_encoder, src_encoder2, src_encoder3, src_encoder4, src_encoder5, src_encoder6, src_classifier, src_data_loader, src_data_loader_eval,src_data_loader2, 
        src_data_loader_eval2, src_data_loader3, src_data_loader_eval3, src_data_loader4, src_data_loader_eval4, src_data_loader5, src_data_loader_eval5, 
        src_data_loader6, src_data_loader_eval6,tgt_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")

    

    eval_src(src_encoder, src_encoder2, src_encoder3, src_encoder4, src_encoder5, src_encoder6,
             src_classifier, src_data_loader_eval, src_data_loader_eval2, src_data_loader_eval3,
             src_data_loader_eval4, src_data_loader_eval5, src_data_loader_eval6)

    
    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    if not (tgt_encoder.restored and critic.restored and
            param.tgt_model_trained):
        tgt_encoder = train_tgt(args, src_encoder, src_encoder2, src_encoder3, src_encoder4, src_encoder5, src_encoder6, tgt_encoder, critic,
                                src_data_loader, src_data_loader2, src_data_loader3, src_data_loader4, src_data_loader5, src_data_loader6, tgt_data_loader)

    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only eval <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval,tgt_test_oneraw,tgt_test_tworaw)
    print(">>> domain adaption eval <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval,tgt_test_oneraw,tgt_test_tworaw)
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader,tgt_train_oneraw,tgt_train_tworaw)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader,tgt_train_oneraw,tgt_train_tworaw)
