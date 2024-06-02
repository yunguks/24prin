import torch
import torchvision
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
from rdkit import Chem
import torch.nn.functional as F
from itertools import chain
from models.model import RNNDecoder, CNNEncoder, RNNDecoderWithATT
import argparse
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def manual_seed(seed):
    np.random.seed(seed) #1
    random.seed(seed) #2
    torch.manual_seed(seed) #3
    torch.cuda.manual_seed(seed) #4.1
    torch.cuda.manual_seed_all(seed) #4.2
    torch.backends.cudnn.benchmark = False #5 
    torch.backends.cudnn.deterministic = True #6
    

#######################################SMILES Tokenizer#######################################
class SMILES_Tokenizer():
    def __init__(self, max_length):
        self.txt2idx = {}
        self.idx2txt = {}
        self.max_length = max_length
    
    def fit(self, SMILES_list): # SMILES_list에서 모든
        unique_char = set()
        for smiles in SMILES_list:
            for char in smiles:
                unique_char.add(char)
        unique_char = sorted(list(unique_char))
        for i, char in enumerate(unique_char): # 0: pad, 1: start,  from 2: ~ , voca_size+1 : end
            self.txt2idx[char]=i+2
            self.idx2txt[i+2]=char
            
        self.voca_size = len(unique_char)
        
    def txt2seq(self, texts):
        seqs = []
        for text in texts:
            seq = [0]*self.max_length
            seq[0] = 1
            for i, t in enumerate(text):
                if i == self.max_length-1:
                    break
                try:
                    seq[i+1] = self.txt2idx[t]
                except:
                    seq[i+1] = 0
            seq[i+2] = 2 + self.voca_size
            seqs.append(seq)
        return np.array(seqs)
    
    def seq2txt(self, seqs):
        return ''.join([self.idx2txt.get(idx, '') for idx in seqs if idx != 0])
        

###################################SMILES DATASET##################################################
class SMILES_dataset(Dataset):
    
    TRANSFROM = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ##  mean, std
    ])
    
    def __init__(self, data_path, tokenizer, train=True, max_length = 167, feature=False):
        
        self.train = train
        if train:
            file_name = 'train_100.csv'
        else:
            file_name = 'val_100.csv'
    
        data = pd.read_csv(os.path.join(data_path, file_name))
        data = data.dropna(subset=['pIC50'])
        
        self.tokenizer = tokenizer
        
        self.feature = feature
        if feature:
            self.imgs = (data_path+'/feature/'+data.file_name).to_numpy()
            for i in range(len(self.imgs)):
                self.imgs[i] = self.imgs[i].replace('.png', '.pt')
        else:    
            self.imgs = (data_path+'/image/'+data.file_name).to_numpy()
        self.smiles = self.tokenizer.txt2seq(data.SMILES)
        #self.labels = data[['pIC50', 'num_atoms', 'logP']].to_numpy()
        self.labels = data[['logP']].to_numpy()
        
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        if self.feature:
            img = torch.load(self.imgs[i], map_location='cpu')
        else:
            img = Image.open(self.imgs[i])
            img = self.TRANSFROM(img)
        
        return {
            'img' : img,
            'smile' : torch.tensor(self.smiles[i], dtype=torch.long),
            'label' : torch.tensor(self.labels[i], dtype=torch.float32)
        }

def get_args():
    parser = argparse.ArgumentParser(description="Smiles training")
    
    ## model
    parser.add_argument('--att', action='store_true', help='Using attetion module')
    parser.add_argument('--teacher', action='store_true', help='Using teacher forcing')
    parser.add_argument('--feature', action='store_true', help='Using Wide-ResNet101 feature extract')
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--enc_dim', type=int, default=512)
    parser.add_argument('--coin', action='store_true', help='Using coin flip in teacher forcing')
    
    # train
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_len', type=int, default=102)
    parser.add_argument('--epoch', type=int, default=100)
    
    # data
    parser.add_argument('--data_path', type=str, default='./kaggle_data')
    
    # util
    parser.add_argument('--save_pth', type=str, default='./checkpoint')
    parser.add_argument('--result_path', type=str, default='result')
    
    
    return parser.parse_args()

def is_smiles(sequence):
    """
    check the sequence matches with the SMILES format
    :param sequence: decoded sequence
    :return: True or False
    """
    m = Chem.MolFromSmiles(sequence)
    return False if m == None else True

if __name__ == "__main__":
    
    args = get_args()
    print(args)
    
    manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    BATCH_SIZE = args.batch
    MODEL_PRETRAIN = True
    EPOCHS = args.epoch
    NUM_LAYERS = 1
    dropout_rate = 0.3
    EMBEDDING_DIM = args.emb_dim
    ENCODER_DIM = args.enc_dim
    LR = args.lr
    DATA_PATH = args.data_path
    TRAIN_CSV_PATH = os.path.join(DATA_PATH, "train_100.csv")
    MAX_LEN = args.max_len
    SAVE_PATH = args.save_pth
    FEATURE = args.feature
    
    save_name = 'result'
    if args.att:
        save_name +='_att'
    if args.teacher:
        save_name +='_teacher'
    if args.feature:
        save_name +='_feature'
    if args.coin:
        save_name +='_coin'
    
    ### make tokenizer
    train_pd = pd.read_csv(TRAIN_CSV_PATH)
    tokenizer = SMILES_Tokenizer(MAX_LEN)
    tokenizer.fit(train_pd.SMILES)
    print(tokenizer.txt2idx.keys())
    voca_size = len(tokenizer.idx2txt)+3
    print(f"voca_size : {voca_size}")
    
    # dataset
    train_dataset = SMILES_dataset(DATA_PATH, tokenizer, train=True, max_length=MAX_LEN, feature=FEATURE)
    print(f"Train data size : {len(train_dataset)}")
    val_dataset = SMILES_dataset(DATA_PATH, tokenizer, train=False, max_length=MAX_LEN, feature=FEATURE)
    print(f"Validation data size : {len(val_dataset)}")
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= BATCH_SIZE, shuffle=False, drop_last=True)
    
    # model
    encoder = CNNEncoder(encoder_dim=ENCODER_DIM, rate=dropout_rate, feature=FEATURE, max_len=MAX_LEN)
    if args.att:
        print("Using Attention Module")
        decoder = RNNDecoderWithATT
    else:
        decoder = RNNDecoder
    decoder = decoder(voca_size=voca_size, embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN, num_layers=NUM_LAYERS, rate=dropout_rate, encoder_dim=ENCODER_DIM)
    
    # # optimizer and loss
    combined_parameters = chain(encoder.parameters(), decoder.parameters())
                
    optimizer = torch.optim.Adam(combined_parameters, lr=LR)
    
    smiles_criterion = nn.CrossEntropyLoss()
    labels_criterion = nn.MSELoss()
    
    # train
    encoder.to(device)
    decoder.to(device)
    best_acc = 0.
    train_acc_per_epoch = []
    train_loss_per_epoch = []
    val_acc_per_epoch = []
    val_loss_per_epoch = []
    for epoch in tqdm(range(EPOCHS), desc="Epoch", position=0, leave=False, total=EPOCHS):
        ## train
        encoder.train()
        decoder.train()
        
        smiles_loss = 0.
        train_count = 0
        train_loss = 0.
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train step", position=1, leave=False)
        for step, batch in progress_bar:
            imgs = batch['img'].to(device)
                
            smiles = batch['smile'].to(device)
            labels = batch['label'].to(device)
            
            dec_in = smiles
            
            enc_out = encoder(imgs)
            smiles_outs, atts  = decoder(enc_out, dec_in,training=False)
            
            smiles_loss = smiles_criterion(smiles_outs.view(-1, voca_size), smiles.view(-1))
            
            labels_loss = torch.tensor(0.)
            
            loss = smiles_loss
            # Add doubly stochastic attention regularization
            loss += ((1. - atts.sum(dim=1)) ** 2).mean()
        
            train_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            preds_output = torch.argmax(smiles_outs, dim=-1)
            for i in range(BATCH_SIZE):
                if torch.equal(preds_output[i], smiles[i]):
                    train_count +=1
                        
            progress_bar.set_postfix(
                {
                "loss": f'{loss.detach().item():.4f}', 
                "smiles": f'{smiles_loss.detach().item():.4f}', 
                "labels": f'{labels_loss.detach().item():.4f}', 
                }
            )
            
        train_acc = train_count / len(train_dataset) * 100
        train_loss = train_loss / len(train_loader)
        train_acc_per_epoch.append(train_acc)
        train_loss_per_epoch.append(train_loss)
        
        print(f"Train Smiles acc : {train_acc:.2f}% , {train_count}")
        ## evalutate
        encoder.eval()
        decoder.eval()
        
        with torch.no_grad():
            # val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Val step", position=1, leave=False)
            smiles_count = 0
            val_loss = 0.
            for batch in tqdm(val_loader, leave=False):
                imgs = batch['img'].to(device)
                smiles = batch['smile'].to(device)
                labels = batch['label'].to(device)
                
                dec_in = torch.ones((BATCH_SIZE,MAX_LEN), dtype=torch.long).to(device)
                
                enc_out = encoder(imgs)
                smiles_outs, alphas  = decoder(enc_out, dec_in, training=False)
            
                
                smiles_loss = smiles_criterion(smiles_outs.view(-1, voca_size), smiles.view(-1))
                # labels_loss = labels_criterion(labels, labels_out)
                
                loss = smiles_loss
                loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
                val_loss += loss.detach().item()
                preds_output = torch.argmax(smiles_outs, dim=-1)
                
                for i in range(BATCH_SIZE):
                    if torch.equal(preds_output[i], smiles[i]):
                        smiles_count+=1
                
                # smiles_count += torch.sum(torch.tensor([torch.equal(preds_output[i], smiles[i]) for i in range(BATCH_SIZE)])).item()
                # for i in range(BATCH_SIZE):
                #     if torch.equal(preds_output[i], smiles[i]):
                #         smiles_acc +=1
            
            smiles_acc = smiles_count / len(val_dataset) * 100
            val_loss = val_loss / len(train_loader)
            val_loss_per_epoch.append(val_loss)
            val_acc_per_epoch.append(smiles_acc)
            print(f"Validation Smiles acc : {smiles_acc:.2f}% , {smiles_count}")
            # best model save
            if not os.path.exists('./checkpoint'):
                os.mkdir('./checkpoint')
            if best_acc < smiles_acc:
                best_acc = smiles_acc
                save_checkpoint = {
                        'encoder' : encoder,
                        'encoder_state_dict' : encoder.state_dict(),
                        'decoder' : decoder,
                        'decoder_state_dict' : decoder.state_dict(),
                        'epoch' : epoch
                    }
                torch.save(save_checkpoint, f'{os.path.join(args.save_pth, save_name)+"_best.pth"}')
                print(f"save best model.")
    
    # last model save
    save_checkpoint = {
            'encoder' : encoder,
            'encoder_state_dict' : encoder.state_dict(),
            'decoder' : decoder,
            'decoder_state_dict' : decoder.state_dict(),
            'epoch' : epoch
        }
    torch.save(save_checkpoint,f'{os.path.join(args.save_pth, save_name)+"_last.pth"}')
    print(f"save last model.")  
    
    result = {
        'train_acc' : train_acc_per_epoch,
        'val_acc' : val_acc_per_epoch,
        'train_loss' : train_loss_per_epoch,
        'val_loss' : val_loss_per_epoch
    }
    result_path = args.result_path
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    result_name = save_name + '.csv'
    result_name = os.path.join(result_path, result_name)
    
    dataframe = pd.DataFrame(result)
    with open(result_name, 'a') as file:
        for i in range(len(train_acc_per_epoch)):
            file.write(f'{i},{train_acc_per_epoch[i]},{val_acc_per_epoch[i]},{train_loss_per_epoch[i]}\n')